# YOLO Training Guide For This ANPR Project

This guide explains the project in simple language and shows how to train your
own YOLO plate detector using a Kaggle Indian number plate dataset.

## What This Project Does

The project has 4 main parts:

```text
Vehicle image/video
        ↓
YOLO detects the number plate box
        ↓
OpenCV cleans the cropped plate image
        ↓
EasyOCR reads the plate text
        ↓
Indian plate rules clean and validate the OCR text
```

YOLO and OCR are different:

- **YOLO** finds the plate location. It outputs a box.
- **OCR** reads the text inside that box.
- **OpenCV** improves the plate crop before OCR.
- **plate_utils.py** fixes and validates Indian plate formats.

Training YOLO improves only the first part: **plate detection**.

If YOLO cannot find the plate, train YOLO.
If YOLO finds the plate but text is wrong, improve OCR/preprocessing.

## Do We Need A Fresh Start?

No. Keep the current app.

We only add one new ML training pipeline:

```text
Kaggle dataset
        ↓
prepare_yolo_dataset.py
        ↓
YOLO-format train/val folders
        ↓
train_yolo_plate.py
        ↓
new best.pt
        ↓
app.py uses the new model
```

## Step 1: Download Dataset From Kaggle

Download one Indian number plate dataset manually from Kaggle.

Recommended first:

```text
https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset
```

Put the extracted dataset here:

```text
datasets/raw_indian_vehicle/
```

If you prefer learning step by step, open:

```text
ANPR_YOLO_Training_Walkthrough.ipynb
```

The notebook runs the same prepare/train/validate flow, but with explanations
between each step.

Your folder may look different depending on the Kaggle zip. That is okay. The
prepare script searches recursively for images and annotations.

## Step 2: Prepare YOLO Dataset

Run:

```bash
python3 scripts/prepare_yolo_dataset.py datasets/raw_indian_vehicle --out datasets/plate_yolo
```

What this does:

- finds images
- finds matching annotations
- converts boxes into YOLO format
- creates train/validation folders
- writes `dataset.yaml` for YOLO training

Output:

```text
datasets/plate_yolo/
  images/
    train/
    val/
  labels/
    train/
    val/
  dataset.yaml
```

## Step 3: Train YOLO

Run this for a small first training run:

```bash
python3 scripts/train_yolo_plate.py --data datasets/plate_yolo/dataset.yaml --epochs 30 --model yolo11n.pt
```

If your machine is slow, use:

```bash
python3 scripts/train_yolo_plate.py --data datasets/plate_yolo/dataset.yaml --epochs 10 --model yolo11n.pt
```

What this does:

- starts from a small YOLO model
- learns to detect one class: `plate`
- saves training results under `runs/detect/plate_train`

Important output:

```text
runs/detect/plate_train/weights/best.pt
```

That is your newly trained detector.

## Step 4: Replace App Model

Back up current model:

```bash
cp best.pt best_old.pt
```

Use your trained model:

```bash
cp runs/detect/plate_train/weights/best.pt best.pt
```

Now your existing Streamlit app uses your trained model automatically.

## Step 5: Run App

```bash
streamlit run app.py
```

Upload images and check:

- Is the plate box appearing more often?
- Are bad/no-detection cases improved?
- Is OCR still wrong sometimes?

If box improves but text is wrong, YOLO training worked; OCR needs tuning next.

## Step 6: Benchmark

After training, run benchmark on labeled images:

```bash
python3 scripts/benchmark_anpr.py failure_samples --labels failure_samples/labels.csv
```

Then update `RESULTS.md` with real numbers.

## Resume-Safe Claim After You Train

Only after you complete training and keep results, you can say:

> Fine-tuned a YOLO plate detector on a labeled Indian number plate dataset and integrated it with OpenCV preprocessing, EasyOCR recognition, and Streamlit deployment.

If you have metrics from YOLO training, you can add:

> Evaluated detection performance using YOLO validation metrics and benchmarked the full ANPR pipeline on a manually labeled test set.

Do not mention mAP, FPS, or accuracy until you have the actual values from your
training and benchmark outputs.

## Interview Explanation

Short version:

> I trained YOLO only for plate localization. YOLO gives the bounding box. Then I crop that region, use OpenCV to improve contrast and sharpness, use EasyOCR to read the characters, and finally apply Indian plate-format rules to clean invalid OCR outputs.

If asked why OCR is separate:

> YOLO is an object detector, not a text reader. It can find where the plate is, but reading characters is an OCR task.

If asked what went wrong:

> I separated errors into detection failures and OCR failures. Detection failures require YOLO fine-tuning. OCR failures require preprocessing and post-processing improvements.
