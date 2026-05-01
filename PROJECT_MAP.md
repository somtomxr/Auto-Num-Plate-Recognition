# Project Map

This file explains what each important file/folder does, so you can understand
and explain the project in interviews.

## Big Picture

```text
app.py
  loads best.pt
  uploads image/video
  sends frame to YOLO
        ↓
ocr_engine.py
  crops detected plate
  preprocesses with OpenCV
  runs EasyOCR/Tesseract
        ↓
plate_utils.py
  cleans OCR mistakes
  validates Indian plate format
        ↓
Streamlit UI
  shows annotated result
  logs plate
  exports CSV
```

## Core App Files

| Path | Why it exists |
| --- | --- |
| `app.py` | Streamlit web app. Handles uploads, UI, YOLO detection, app state, logs, CSV export. |
| `ocr_engine.py` | OCR pipeline. Takes a plate crop and returns detected text. Uses OpenCV preprocessing + OCR candidates + scoring. |
| `plate_utils.py` | Indian plate cleanup and validation. Handles state codes, BH series, formatting, and OCR character correction. |
| `best.pt` | YOLO model weights loaded by the app. This detects plate bounding boxes. |

## Training And Evaluation

| Path | Why it exists |
| --- | --- |
| `TRAINING_GUIDE.md` | Step-by-step guide for transfer learning / fine-tuning YOLO on Kaggle data. |
| `ANPR_YOLO_Training_Walkthrough.ipynb` | Notebook-first training walkthrough. Best file for learning every step interactively. |
| `scripts/prepare_yolo_dataset.py` | Converts raw Kaggle annotations into YOLO format. |
| `scripts/train_yolo_plate.py` | Trains/fine-tunes YOLO for one class: `plate`. |
| `scripts/benchmark_anpr.py` | Runs the full app pipeline on a folder of images and writes measured results. |
| `RESULTS.md` | Place for honest benchmark/training results after you run them. |
| `failure_samples/` | Manual examples where detection/OCR works or fails. Used for error analysis. |
| `datasets/` | Local Kaggle downloads and prepared YOLO datasets. Ignored by git because it can be large. |
| `runs/` | YOLO training outputs. Ignored by git because it can be large. |

## Learning / Interview Files

| Path | Why it exists |
| --- | --- |
| `INTERVIEW_GUIDE.md` | Simple answers for explaining YOLO, OCR, transfer learning, failures, and improvements. |
| `Som_Tomar_Resume_Draft.md` | Two-project resume draft with classical ML kept in the Skills section. |
| `ARCHITECTURE.md` | Deeper technical explanation of the ANPR pipeline. |
| `README.md` | Main public project documentation. |
| `QUICKSTART.md` | Quick setup and run instructions. |

## Tests

| Path | Why it exists |
| --- | --- |
| `tests/test_plate_utils.py` | Unit tests for Indian plate validation/formatting/cleanup. |

Run tests:

```bash
python3 -m pytest tests/test_plate_utils.py -q
```

## What To Track In Git

Track:

- code
- guides
- tests
- small templates
- final resume draft

Do not track:

- Kaggle dataset files
- YOLO training runs
- benchmark CSV/JSON outputs unless you intentionally want to publish a small result
- personal failure sample images

## Interview-Friendly Project Story

> I built an end-to-end Indian ANPR app. YOLO detects the plate location,
> OpenCV improves the crop, OCR reads the text, and custom Python logic validates
> Indian plate formats. I then added a transfer-learning path to fine-tune YOLO
> on Kaggle Indian plate data and benchmark the complete pipeline.
