"""Train and evaluate a room condition classifier."""

import argparse
from itertools import batched
import json
from pathlib import Path
import random
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torchvision.transforms import v2
from transformers import SiglipImageProcessor, SiglipVisionModel

from predict import IMAGE_EXTENSIONS, LABELS, MODEL_DIR, ROOT, load_encoder, read_images

type Features = NDArray[np.float32]
type Probabilities = NDArray[np.float64]
type Labels = NDArray[np.int64]


class Scores(NamedTuple):
    accuracy: float
    macro_f1: float
    log_loss: float


def extract_features(
    paths: list[Path],
    encoder: SiglipVisionModel,
    processor: SiglipImageProcessor,
    *,
    augment: bool = False,
) -> Features:
    """Return features with shape [view, image, feature]. View zero is the original image."""
    random_state = random.Random(42)
    torch.manual_seed(42)
    jitter = v2.ColorJitter(brightness=0.1, contrast=0.1)
    mean = torch.tensor(processor.image_mean)[:, None, None]
    std = torch.tensor(processor.image_std)[:, None, None]
    features: list[Features] = []
    with torch.inference_mode():
        for batch in batched(paths, 4):
            pixels = read_images(batch, processor)
            views = [pixels]
            if augment:
                augmented = []
                for image in pixels:
                    if random_state.random() < 0.5:
                        image = image.flip(-1)
                    image = (image * std + mean).clamp(0, 1)
                    augmented.append((jitter(image) - mean) / std)
                views.append(torch.stack(augmented))
            features.append(
                np.stack(
                    [
                        encoder(pixel_values=view.to(encoder.device))
                        .pooler_output.cpu()
                        .numpy()
                        for view in views
                    ],
                    axis=0,
                )
            )
    return np.concatenate(features, axis=1)


def fit_classifier(features: Features, labels: Labels) -> LogisticRegression:
    views = features.shape[0]
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(
        features.reshape(-1, features.shape[-1]).astype(np.float64),
        np.tile(labels, views),
        sample_weight=np.full(len(labels) * views, 1.0 / views),
    )


def score(labels: Labels, probabilities: Probabilities) -> Scores:
    predicted = probabilities.argmax(axis=1)
    return Scores(
        float(accuracy_score(labels, predicted)),
        float(f1_score(labels, predicted, average="macro")),
        float(log_loss(labels, probabilities, labels=range(len(LABELS)))),
    )


def evaluate_folds(
    features: Features,
    labels: Labels,
) -> tuple[Probabilities, list[tuple[Scores, Scores]]]:
    probabilities = np.empty((len(labels), len(LABELS)))
    scores = []
    for training, validation in StratifiedKFold(5, shuffle=True, random_state=42).split(
        features[0], labels
    ):
        classifier = fit_classifier(features[:, training], labels[training])
        probabilities[validation] = classifier.predict_proba(features[0, validation])
        scores.append(
            (
                score(
                    labels[training], classifier.predict_proba(features[0, training])
                ),
                score(labels[validation], probabilities[validation]),
            )
        )
    return probabilities, scores


def plot_results(
    labels: Labels,
    probabilities: Probabilities,
    holdout: NDArray[np.bool_],
    folds: list[tuple[Scores, Scores]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for axis, selection, title in zip(
        axes[0],
        (~holdout, holdout),
        ("Cross-validation", "Evaluation"),
    ):
        ConfusionMatrixDisplay.from_predictions(
            labels[selection],
            probabilities[selection].argmax(axis=1),
            labels=range(len(LABELS)),
            display_labels=LABELS,
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(title)
    numbers = range(1, len(folds) + 1)
    for scores, name in (
        ([training for training, _ in folds], "Training"),
        ([validation for _, validation in folds], "Validation"),
    ):
        axes[1, 0].plot(
            numbers, [s.accuracy for s in scores], "o-", label=f"{name} accuracy"
        )
        axes[1, 1].plot(numbers, [s.log_loss for s in scores], "o-", label=name)
    axes[1, 0].plot(
        numbers, [s.macro_f1 for _, s in folds], "o-", label="Validation macro F1"
    )
    axes[1, 0].set(title="Scores by fold", ylim=(0, 1))
    axes[1, 1].set(title="Log loss by fold", ylabel="Log loss")
    for axis in axes[1]:
        axis.set(xticks=numbers, xlabel="Fold")
        axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def train(image_folder: Path) -> Scores:
    torch.set_num_threads(8)
    paths = [
        path
        for label in LABELS
        for path in sorted((image_folder / label).iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = np.array(
        [LABELS.index(path.parent.name) for path in paths], dtype=np.int64
    )
    training, holdout = train_test_split(
        np.arange(len(paths)),
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    encoder, processor = load_encoder()
    features = extract_features(
        [paths[i] for i in training], encoder, processor, augment=True
    )
    print(f"Extracted features from {len(training)} training images.", flush=True)
    probabilities, folds = evaluate_folds(features, labels[training])
    classifier = fit_classifier(features, labels[training])
    holdout_features = extract_features(
        [paths[i] for i in holdout], encoder, processor
    )[0]
    holdout_probabilities = classifier.predict_proba(holdout_features)
    result = score(labels[holdout], holdout_probabilities)
    torch.save(
        {
            "weight": torch.from_numpy(classifier.coef_).float(),
            "bias": torch.from_numpy(classifier.intercept_).float(),
        },
        MODEL_DIR / "head.pt",
    )
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    metrics = {
        "cross_validation": score(labels[training], probabilities)._asdict(),
        "holdout": result._asdict(),
        "folds": [
            {"training": t._asdict(), "validation": v._asdict()} for t, v in folds
        ],
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_results(
        labels[np.concatenate((training, holdout))],
        np.concatenate((probabilities, holdout_probabilities)),
        np.arange(len(paths)) >= len(training),
        folds,
        reports / "evaluation.png",
    )
    print(
        classification_report(
            labels[holdout],
            holdout_probabilities.argmax(axis=1),
            target_names=LABELS,
            zero_division=0,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_folder", type=Path)
    args = parser.parse_args()
    result = train(args.image_folder)
    print(
        f"Evaluation accuracy: {result.accuracy:.2%}. Macro F1: {result.macro_f1:.4f}."
    )


if __name__ == "__main__":
    main()
