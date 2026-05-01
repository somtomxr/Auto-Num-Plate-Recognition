"""Benchmark the ANPR pipeline on a local folder of images.

Optional labels CSV format:
    filename,plate
    car_001.jpg,MH12AB1234

The script writes a per-image CSV and a summary JSON. Use the generated
summary to update RESULTS.md and resume metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_engine import read_plate  # noqa: E402
from plate_utils import format_plate, is_valid_indian_plate  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    labels: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "filename" not in reader.fieldnames or "plate" not in reader.fieldnames:
            raise ValueError("Labels CSV must include filename and plate columns.")
        for row in reader:
            labels[row["filename"]] = row["plate"].upper().replace(" ", "")
    return labels


def _iter_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _padded_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(int, xyxy)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_left = max(2, int(bw * 0.05))
    pad_right = max(3, int(bw * 0.14))
    pad_top = max(2, int(bh * 0.12))
    pad_bottom = max(2, int(bh * 0.14))
    x1, y1 = max(0, x1 - pad_left), max(0, y1 - pad_top)
    x2, y2 = min(frame.shape[1], x2 + pad_right), min(frame.shape[0], y2 + pad_bottom)
    return frame[y1:y2, x1:x2]


def benchmark(args: argparse.Namespace) -> tuple[list[dict], dict]:
    image_paths = _iter_images(args.input_dir)
    labels = _load_labels(args.labels)

    yolo = YOLO(str(args.model))
    reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)

    rows: list[dict] = []
    total_ms = 0.0

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            rows.append({
                "filename": image_path.name,
                "status": "unreadable",
                "expected": labels.get(image_path.name, ""),
                "prediction": "",
                "formatted": "",
                "confidence": 0.0,
                "valid": False,
                "exact_match": False,
                "elapsed_ms": 0.0,
            })
            continue

        start = time.perf_counter()
        detections = yolo(
            frame,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            imgsz=args.imgsz,
            verbose=False,
        )[0]

        best_text = ""
        best_conf = 0.0
        for box in detections.boxes:
            crop = _padded_crop(frame, box.xyxy[0].cpu().numpy())
            if crop.size == 0:
                continue
            result = read_plate(crop, reader, use_tesseract=args.tesseract)
            if result["confidence"] > best_conf:
                best_text = result["text"]
                best_conf = result["confidence"]

        elapsed_ms = (time.perf_counter() - start) * 1000
        total_ms += elapsed_ms

        expected = labels.get(image_path.name, "")
        exact_match = bool(expected and best_text == expected)
        rows.append({
            "filename": image_path.name,
            "status": "ok",
            "expected": expected,
            "prediction": best_text,
            "formatted": format_plate(best_text) if best_text else "",
            "confidence": round(best_conf, 4),
            "valid": is_valid_indian_plate(best_text) if best_text else False,
            "exact_match": exact_match,
            "elapsed_ms": round(elapsed_ms, 2),
        })

    readable = [r for r in rows if r["status"] == "ok"]
    predicted = [r for r in readable if r["prediction"]]
    labelled = [r for r in readable if r["expected"]]
    valid = [r for r in predicted if r["valid"]]
    exact = [r for r in labelled if r["exact_match"]]

    summary = {
        "image_count": len(image_paths),
        "readable_images": len(readable),
        "plates_predicted": len(predicted),
        "valid_predictions": len(valid),
        "labelled_images": len(labelled),
        "exact_matches": len(exact),
        "avg_ms_per_readable_image": round(total_ms / len(readable), 2) if readable else 0.0,
        "settings": {
            "model": str(args.model),
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "max_det": args.max_det,
            "gpu": args.gpu,
            "tesseract": args.tesseract,
        },
    }
    if predicted:
        summary["valid_prediction_rate"] = round(len(valid) / len(predicted), 4)
    if labelled:
        summary["exact_match_rate"] = round(len(exact) / len(labelled), 4)

    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ANPR on local images.")
    parser.add_argument("input_dir", type=Path, help="Folder containing test images.")
    parser.add_argument("--labels", type=Path, help="Optional CSV with filename,plate columns.")
    parser.add_argument("--model", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evaluation")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=15)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--tesseract", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = benchmark(args)

    csv_path = args.out_dir / "benchmark_predictions.csv"
    json_path = args.out_dir / "benchmark_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "filename",
            "status",
            "expected",
            "prediction",
            "formatted",
            "confidence",
            "valid",
            "exact_match",
            "elapsed_ms",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
