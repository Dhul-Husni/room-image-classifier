# Room condition classification

Download and extract [submission.zip](https://github.com/Dhul-Husni/room-image-classifier/releases/latest/download/submission.zip).

## Inference

```bash
uv sync --locked
uv run python predict.py /path/to/photos --output predictions.csv
```

Input: JPEG, PNG, or WebP images in a folder and its subfolders.

Output: a CSV file with `filename,label` columns. Labels are **dated**, **distressed**, or **renovated**.

## Training

Input: the image folder with `dated/`, `distressed/`, and `renovated/` subfolders.

```bash
uv run python train.py /path/to/labelled
```

Output: the classifier in `model/head.pt` and evaluation results and graphs in `reports/`.

Training uses a stratified 80/20 split and five training folds, with random seed 42.
