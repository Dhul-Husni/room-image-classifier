"""Assign a room condition label to each image in a folder."""

import argparse
from collections.abc import Sequence
import csv
from itertools import batched
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch import Tensor, nn
from transformers import SiglipImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
LABELS = ("dated", "distressed", "renovated")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def load_encoder() -> tuple[SiglipVisionModel, SiglipImageProcessor]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    directory = MODEL_DIR / "encoder"
    processor = SiglipImageProcessor.from_pretrained(directory, local_files_only=True)
    encoder = SiglipVisionModel.from_pretrained(directory, local_files_only=True)
    encoder.to(device)  # type: ignore[arg-type]  # Incorrect bound-method annotation in Transformers.
    encoder.requires_grad_(False)
    encoder.eval()  # type: ignore[no-untyped-call]  # Transformers does not annotate this method.
    return encoder, processor


def read_images(paths: Sequence[Path], processor: SiglipImageProcessor) -> Tensor:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    pixels: Tensor = processor(images=images, return_tensors="pt").pixel_values
    return pixels


def predict_folder(image_folder: Path, output_csv: Path) -> int:
    paths = sorted(
        path
        for path in image_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No JPEG, PNG, or WebP images in {image_folder}")

    encoder, processor = load_encoder()
    head = nn.Linear(encoder.config.hidden_size, len(LABELS)).to(encoder.device)
    head.load_state_dict(
        torch.load(
            MODEL_DIR / "head.pt", map_location=encoder.device, weights_only=True
        )
    )
    with output_csv.open("w", newline="") as file, torch.inference_mode():
        writer = csv.writer(file)
        writer.writerow(("filename", "label"))
        for batch in batched(paths, 4):
            pixels = read_images(batch, processor).to(encoder.device)
            features = encoder(pixel_values=pixels).pooler_output
            predicted = head(features).argmax(dim=1).cpu().tolist()
            writer.writerows(
                (path.relative_to(image_folder).as_posix(), LABELS[index])
                for path, index in zip(batch, predicted)
            )
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    count = predict_folder(args.image_folder, args.output)
    print(f"Saved {count} predictions to {args.output}")


if __name__ == "__main__":
    main()
