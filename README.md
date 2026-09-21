# Room condition classification

```bash
git clone https://github.com/Dhul-Husni/room-image-classifier.git
cd room-image-classifier
```

Download [model.zip](https://github.com/Dhul-Husni/room-image-classifier/releases/download/v0.1.0/model.zip) into the cloned repository.

```bash
unzip model.zip
uv sync --locked
```

## Inference

```bash
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
