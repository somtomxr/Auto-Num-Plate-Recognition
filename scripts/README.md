# Scripts Folder

These scripts help you train, evaluate, and debug the ANPR project.

## `prepare_yolo_dataset.py`

Converts raw Kaggle annotations into YOLO format.

```bash
python3 scripts/prepare_yolo_dataset.py datasets/raw_indian_vehicle --out datasets/plate_yolo
```

Use this before training.

## `train_yolo_plate.py`

Fine-tunes a pretrained YOLO model for one class: `plate`.

```bash
python3 scripts/train_yolo_plate.py --data datasets/plate_yolo/dataset.yaml --epochs 30 --model yolo11n.pt
```

This is transfer learning.

## `benchmark_anpr.py`

Runs the full ANPR pipeline on images and optional labels.

```bash
python3 scripts/benchmark_anpr.py failure_samples --labels failure_samples/labels.csv
```

Use this to get resume-safe metrics.

## `analyze_failure_samples.py`

Checks your manually written failure-sample CSV.

```bash
python3 scripts/analyze_failure_samples.py failure_samples/labels.csv
```

Use this before asking Codex to analyze failures.
