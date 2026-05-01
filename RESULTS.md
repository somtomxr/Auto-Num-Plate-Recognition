# ANPR Results

This file is evidence-first. It separates detector training metrics from the
full ANPR pipeline benchmark, because YOLO detection and OCR recognition are
different tasks.

## Current Status

- Pipeline: fine-tuned YOLO/Ultralytics detection, OpenCV preprocessing, EasyOCR OCR, Indian plate cleanup and validation.
- App surface: Streamlit image upload, video processing, annotated output, detection log, and CSV export.
- Validation coverage: unit tests for Indian plate formatting, BH-series handling, noisy OCR cleanup, and invalid state codes.
- YOLO detector training: completed on Google Colab T4 GPU.
- Full ANPR benchmark: ✅ completed on 339 labeled Indian plate images (CPU, macOS).

## YOLO Detector Training

Training objective: detect one class, `plate`.

| Item | Value |
| --- | --- |
| Training method | Transfer learning / fine-tuning |
| Base model | YOLO11n pretrained model |
| Dataset | Kaggle Indian vehicle license plate dataset |
| Annotated images | 1,698 |
| Train split | 1,359 images |
| Validation split | 339 images |
| Epochs | 30 |
| Image size | 640 |
| Batch size | 16 |
| Hardware | Google Colab Tesla T4 GPU |
| Precision | 0.997 |
| Recall | 0.994 |
| mAP@50 | 0.994 |
| mAP@50-95 | 0.867 |

Artifacts saved from Colab:

- `best.pt` downloaded and copied into the local app.
- Previous local model backed up as `best_old.pt`.
- `plate_train_results.zip` downloaded for training proof and visualizations.

Important: these are **plate detection metrics**, not final OCR text accuracy.
OCR quality must be measured separately with the full ANPR benchmark below.

## Full ANPR Pipeline Benchmark

Benchmark run on the same 339-image validation set used for YOLO training.
Each image was run through the complete pipeline: YOLO detection → OpenCV
preprocessing → EasyOCR recognition → Indian plate validation.

| Metric | Value |
| --- | --- |
| Test images | 339 |
| Images with a plate detected | 329 (97.1%) |
| Predictions passing Indian format validation | 217 (66.0%) |
| Exact-match OCR accuracy | 73 / 339 (21.5%) |
| Average processing time per image | ~3,125 ms (CPU) |
| Hardware | Apple Mac CPU (no GPU) |
| EasyOCR GPU | Disabled |
| Tesseract | Enabled (v5.5.2) |

**Interpretation:**

- **97.1% detection rate** — the fine-tuned YOLO model reliably locates plates.
- **66.0% valid-format rate** — OCR output passes Indian plate structure rules (state code, RTO digits, series, number) for nearly two-thirds of images.
- **21.5% exact-match rate** — character-level exact match against ground-truth labels. This improved from 19.5% after adding deskew and Tesseract. Real-world OCR on diverse angles, lighting, and blur makes exact match inherently low without a dedicated OCR model.

Common failure causes: motion blur, glare, angled plates, small/distant plates,
OCR confusion between O/0, I/1, S/5, B/8.

## How To Re-run Benchmark

```bash
python3 scripts/build_val_labels.py   # builds evaluation/val_labels.csv
python3 scripts/benchmark_anpr.py datasets/plate_yolo/images/val \
  --labels evaluation/val_labels.csv --model best.pt
```

Outputs: `evaluation/benchmark_predictions.csv`, `evaluation/benchmark_summary.json`

## Resume-Safe Claim Template

> Fine-tuned a pretrained YOLO11n detector via transfer learning on 1,698 labeled
> Indian vehicle images, achieving 0.994 mAP@50 on a held-out 339-image validation
> split (97.1% plate detection rate). Integrated the model into a Streamlit ANPR
> app with a 5-variant OpenCV preprocessing pipeline (with auto-deskew), dual-engine 
> OCR (EasyOCR + Tesseract), and Indian plate post-processing, benchmarked 
> end-to-end on 339 real-world images achieving a 66.0% valid-format prediction rate.
