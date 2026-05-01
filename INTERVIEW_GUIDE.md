# ANPR Interview Guide

## One-Minute Explanation

I built an Indian Automatic Number Plate Recognition app. The system uses YOLO
to detect the location of the plate in an image or video frame. After detection,
I crop the plate region and use OpenCV preprocessing to improve contrast,
sharpness, thresholding, and noise. Then EasyOCR reads the characters. Finally,
I apply Indian license plate rules to clean common OCR mistakes and validate
formats like standard plates, BH-series plates, and two-line plates. The result
is shown in a Streamlit app with annotated output, confidence, logs, and CSV
export.

## Two-Minute Explanation

The project has two ML/computer-vision stages. First, YOLO is used as an object
detector. It does not read text; it only predicts a bounding box around the
number plate. Second, OCR reads text from that cropped plate image.

I separated the problem this way because detection and recognition fail for
different reasons. If no bounding box appears, the detector needs better data or
fine-tuning. If the box appears but the text is wrong, then the OCR pipeline
needs better preprocessing or post-processing.

The app uses OpenCV to create multiple plate crop variants, including contrast
enhancement, sharpening, Otsu thresholding, adaptive thresholding, and
morphology. These variants help OCR handle different lighting and plate quality.

After OCR, I clean the text using Indian plate structure. For example, the first
two characters of normal plates should be state letters, the next two should be
RTO digits, and the final part usually contains series letters and four digits.
This helps fix OCR confusion such as O/0, I/1, S/5, and B/8.

## If Asked: What Did You Train?

Current safe answer before training:

> The app integrates a YOLO plate detector with OCR and Indian plate validation.
> I am using the detector weights as the plate localization model and evaluating
> the full pipeline on Indian plate samples.

After training on Kaggle:

> I fine-tuned a pretrained YOLO11n model on 1,698 labeled Indian plate images
> for one class: plate. The detector reached 0.994 mAP@50 on a 339-image
> validation split. The trained detector outputs plate bounding boxes, and the
> rest of the pipeline handles OCR and validation.

## If Asked: What Is `best.pt`?

`best.pt` is a PyTorch weights file saved by YOLO training. It contains the
trained detector model. In this app, `app.py` loads `best.pt` and uses it to find
number plate boxes.

## If Asked: Why Use OCR If YOLO Is Already ML?

YOLO is an object detector. It can find the plate region, but it does not read
the characters. OCR is needed to convert the cropped plate image into text.

## If Asked: Why Not Train One Model To Read Everything?

For a fresher project, splitting the problem is simpler and explainable:

- YOLO handles localization.
- OCR handles text recognition.
- Python rules handle Indian plate validation.

This modular design also helps debug errors.

## If Asked: What Are The Main Failure Cases?

- Small or far-away plates
- Motion blur
- Glare or reflections
- Angled plates
- Dirty or damaged plates
- OCR confusion between similar characters like O/0, I/1, S/5, B/8

## If Asked: How Would You Improve It?

- Added deskew preprocessing to auto-correct plate tilt before OCR.
- Enabled Tesseract as a secondary engine alongside EasyOCR.
- Added a REST API with FastAPI so other systems can call the pipeline programmatically.
- Containerized with Docker so it can be deployed on any machine without manual setup.
- Benchmarked on 339 labeled images — 97.1% detection rate, 62.8% valid-format rate.
- Next: try PaddleOCR or a plate-specific OCR model to improve character accuracy.

## Current Training Result

- Base model: YOLO11n pretrained model
- Method: transfer learning / fine-tuning
- Dataset: Kaggle Indian vehicle license plate dataset
- Annotated images: 1,698
- Train split: 1,359 images
- Validation split: 339 images
- Epochs: 30
- Hardware: Google Colab Tesla T4 GPU
- Precision: 0.997
- Recall: 0.994
- mAP@50: 0.994
- mAP@50-95: 0.867

## Full Pipeline Benchmark

- Test images: 339 real-world Indian plate images
- Plate detection rate: 97.1%
- Valid Indian-format prediction rate: 62.8%
- Speed: ~1.5 s/image on CPU (no GPU)

Important interview note:

> Detection metrics (mAP) prove the model finds plates. The OCR benchmark proves
> the full pipeline works end-to-end. Both are different, and I measured both.

## If Asked: What Is The REST API?

I added a FastAPI wrapper around the ANPR pipeline. It exposes three endpoints:

- `GET  /health` — checks if the API and model are loaded and ready
- `POST /predict` — accepts a single image upload, returns plate text as JSON
- `POST /predict/batch` — accepts up to 10 images, returns a list of results

This means another app, mobile frontend, or service can call the ANPR system over
HTTP without needing Python or the Streamlit UI.

Example response from `/predict`:

```json
{
  "success": true,
  "plate_detected": true,
  "result": {
    "plate_text": "MH12AB1234",
    "confidence": 0.87,
    "valid_indian_format": true,
    "two_line_plate": false,
    "detection_confidence": 0.94,
    "processing_time_ms": 1240.5
  }
}
```

To run the API locally:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs for interactive Swagger UI
```

## If Asked: What Is Docker / Why Did You Containerize It?

Docker packages the entire app — Python version, Tesseract binary, all pip
packages, and the model weights — into a single container image. Anyone can
run it on any machine with one command:

```bash
docker-compose up
# Streamlit → http://localhost:8501
# FastAPI   → http://localhost:8000/docs
```

Without Docker, a new machine needs: Python install, Homebrew, Tesseract,
pip install of 15+ packages, and the correct versions. Docker eliminates all
of that setup. It also makes cloud deployment straightforward.

## If Asked: What Is Deskew?

Deskew is automatic rotation correction. If a photo is taken at a slight angle,
the plate text appears tilted. OCR accuracy drops significantly on tilted text.

The `_deskew` function:
1. Binarizes the crop to find text pixel coordinates
2. Fits a minimum bounding rectangle to those pixels
3. Detects the tilt angle (limited to ±15° to avoid false corrections)
4. Rotates the image back to horizontal using `cv2.warpAffine`

All 5 preprocessing variants then get a straightened input image, which helps
both EasyOCR and Tesseract read characters more accurately.

