"""
FastAPI REST API for the Indian ANPR Pipeline.

Endpoints:
    GET  /              → redirect to /docs
    GET  /health        → liveness check
    POST /predict       → upload image, get plate text as JSON
    POST /predict/batch → upload multiple images, get list of results

Usage (local):
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Usage (Docker):
    Handled automatically by docker-compose.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_engine import read_plate  # noqa: E402
from plate_utils import is_valid_indian_plate  # noqa: E402

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Indian ANPR API",
    description=(
        "Automatic Number Plate Recognition for Indian vehicles.\n\n"
        "Upload an image to detect and read the license plate. "
        "The pipeline uses a fine-tuned YOLO11n detector and EasyOCR with "
        "Indian plate post-processing."
    ),
    version="1.0.0",
    contact={
        "name": "Som Tomar",
        "url": "https://github.com/somtomxr",
    },
)

# ── Model loading (singleton) ─────────────────────────────────────────────────

_MODEL_PATH = ROOT / "best.pt"
_yolo: Optional[YOLO] = None
_reader: Optional[easyocr.Reader] = None


def _get_models() -> tuple[YOLO, easyocr.Reader]:
    """Load models once and cache them for the lifetime of the process."""
    global _yolo, _reader
    if _yolo is None:
        _yolo = YOLO(str(_MODEL_PATH))
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _yolo, _reader


# ── Request / Response schemas ─────────────────────────────────────────────────

class PlateResult(BaseModel):
    plate_text: str
    confidence: float
    valid_indian_format: bool
    two_line_plate: bool
    raw_ocr: str
    detection_confidence: float
    processing_time_ms: float


class PredictResponse(BaseModel):
    success: bool
    plate_detected: bool
    result: Optional[PlateResult] = None
    message: str = ""


class BatchPredictResponse(BaseModel):
    total_images: int
    plates_detected: int
    results: list[dict]


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_path: str


# ── Helper ────────────────────────────────────────────────────────────────────

def _padded_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Crop detected plate with padding to preserve edge characters."""
    x1, y1, x2, y2 = map(int, xyxy)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_l = max(2, int(bw * 0.05))
    pad_r = max(3, int(bw * 0.14))
    pad_t = max(2, int(bh * 0.12))
    pad_b = max(2, int(bh * 0.14))
    x1 = max(0, x1 - pad_l)
    y1 = max(0, y1 - pad_t)
    x2 = min(frame.shape[1], x2 + pad_r)
    y2 = min(frame.shape[0], y2 + pad_b)
    return frame[y1:y2, x1:x2]


def _bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure the file is a valid image.")
    return img


def _run_pipeline(frame: np.ndarray, yolo: YOLO, reader: easyocr.Reader) -> dict:
    """Run YOLO detection + OCR on a single frame."""
    t0 = time.monotonic()

    results = yolo(frame, conf=0.20, iou=0.45, imgsz=640, max_det=15, verbose=False)
    boxes = results[0].boxes if results else None

    if boxes is None or len(boxes) == 0:
        return {"detected": False, "elapsed_ms": (time.monotonic() - t0) * 1000}

    # Use highest confidence detection
    confs = boxes.conf.cpu().numpy()
    best_i = int(confs.argmax())
    det_conf = float(confs[best_i])
    xyxy = boxes.xyxy[best_i].cpu().numpy()
    crop = _padded_crop(frame, xyxy)

    ocr_result = read_plate(crop, reader, use_tesseract=False)
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "detected": True,
        "plate_text": ocr_result["text"],
        "confidence": ocr_result["confidence"],
        "valid_indian_format": is_valid_indian_plate(ocr_result["text"]),
        "two_line_plate": ocr_result["two_line"],
        "raw_ocr": ocr_result["raw"],
        "detection_confidence": round(det_conf, 4),
        "elapsed_ms": round(elapsed, 1),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    Liveness check. Also reports whether the YOLO model file exists.
    Use this endpoint to verify the API is running before sending images.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_MODEL_PATH.exists(),
        model_path=str(_MODEL_PATH),
    )


@app.post("/predict", response_model=PredictResponse, tags=["ANPR"])
async def predict(file: UploadFile = File(..., description="Vehicle image (JPG, PNG, WEBP)")):
    """
    Detect and read the number plate in an uploaded vehicle image.

    **Pipeline:**
    1. YOLO11n detects the plate bounding box
    2. OpenCV crop + deskew + 5-variant preprocessing
    3. EasyOCR reads the text
    4. Indian plate post-processing cleans the result

    **Returns:** plate text, confidence, format validity, and processing time.
    """
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        frame = _bytes_to_bgr(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    yolo, reader = _get_models()

    try:
        out = _run_pipeline(frame, yolo, reader)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    if not out["detected"]:
        return PredictResponse(
            success=True,
            plate_detected=False,
            message="No plate detected in the image.",
        )

    return PredictResponse(
        success=True,
        plate_detected=True,
        result=PlateResult(
            plate_text=out["plate_text"],
            confidence=out["confidence"],
            valid_indian_format=out["valid_indian_format"],
            two_line_plate=out["two_line_plate"],
            raw_ocr=out["raw_ocr"],
            detection_confidence=out["detection_confidence"],
            processing_time_ms=out["elapsed_ms"],
        ),
        message="Plate detected successfully.",
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["ANPR"])
async def predict_batch(files: list[UploadFile] = File(..., description="Multiple vehicle images")):
    """
    Process multiple images in one request.
    Returns a list of results in the same order as the uploaded files.
    Maximum 10 images per request.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch request.")

    yolo, reader = _get_models()
    results = []
    plates_detected = 0

    for f in files:
        raw_bytes = await f.read()
        try:
            frame = _bytes_to_bgr(raw_bytes)
            out = _run_pipeline(frame, yolo, reader)
        except Exception as exc:
            results.append({"filename": f.filename, "error": str(exc)})
            continue

        if out["detected"]:
            plates_detected += 1
            results.append({
                "filename": f.filename,
                "plate_detected": True,
                "plate_text": out["plate_text"],
                "confidence": out["confidence"],
                "valid_indian_format": out["valid_indian_format"],
                "processing_time_ms": out["elapsed_ms"],
            })
        else:
            results.append({"filename": f.filename, "plate_detected": False})

    return BatchPredictResponse(
        total_images=len(files),
        plates_detected=plates_detected,
        results=results,
    )
