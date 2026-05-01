# Failure Samples

Use this folder to collect examples before changing the model or OCR pipeline.

## What To Add

Add 20-30 images into this folder:

- 10 images where the plate is not detected at all
- 10 images where the plate box appears but OCR text is wrong
- 5 images where the app works correctly

Rename images simply:

```text
car_001.jpg
car_002.jpg
car_003.jpg
```

## Fill The CSV

Copy `labels_template.csv` to `labels.csv`, then fill it:

```csv
filename,actual_plate,problem_type,notes
car_001.jpg,MH12AB1234,no_detection,plate visible but YOLO missed it
car_002.jpg,DL01CA5678,wrong_ocr,detected as DL01CA567B
car_003.jpg,UP16AB1234,works,clear image
```

Allowed `problem_type` values:

- `no_detection`
- `wrong_ocr`
- `works`

## Next Command

After adding images and filling `labels.csv`, run:

```bash
python3 scripts/analyze_failure_samples.py failure_samples/labels.csv
```

Then ask Codex:

> I added failure samples, analyze them and tell me whether to tune OCR or train YOLO.
