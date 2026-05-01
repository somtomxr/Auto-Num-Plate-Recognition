"""Build evaluation/val_labels.csv from the val image set.

The prepare_yolo_dataset.py script renamed images like:
    000000_ML7.jpg  ←  original ML7.jpg

We strip the leading 6-digit prefix, find the matching XML annotation
in the raw dataset, and extract the plate text from <object><name>.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "datasets" / "plate_yolo" / "images" / "val"
RAW_DIR = ROOT / "datasets" / "raw_indian_vehicle"
OUT_CSV = ROOT / "evaluation" / "val_labels.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_xml_index(raw_dir: Path) -> dict[str, Path]:
    """Map stem → xml path for every XML in raw_dir (recursive)."""
    index: dict[str, Path] = {}
    for xml_path in raw_dir.rglob("*.xml"):
        index[xml_path.stem.lower()] = xml_path
    return index


def extract_plate_text(xml_path: Path) -> str | None:
    try:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip().upper().replace(" ", "")
            if name:
                return name
    except Exception:
        pass
    return None


def original_stem(val_filename: str) -> str:
    """Strip the 000000_ prefix added during dataset conversion."""
    stem = Path(val_filename).stem
    # Format is NNNNNN_original_stem
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 6:
        return parts[1]
    return stem


def main() -> None:
    xml_index = build_xml_index(RAW_DIR)
    print(f"XML index size: {len(xml_index)}")

    val_images = sorted(
        p for p in VAL_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Val images found: {len(val_images)}")

    rows: list[dict] = []
    matched = 0
    skipped = 0

    for img_path in val_images:
        orig = original_stem(img_path.name)
        orig_lower = orig.lower()

        xml_path = xml_index.get(orig_lower)
        if xml_path is None:
            skipped += 1
            continue

        plate = extract_plate_text(xml_path)
        if not plate:
            skipped += 1
            continue

        rows.append({"filename": img_path.name, "plate": plate})
        matched += 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "plate"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone.")
    print(f"  Matched:  {matched}")
    print(f"  Skipped:  {skipped}")
    print(f"  CSV saved: {OUT_CSV}")

    if rows:
        print("\nFirst 5 rows:")
        for r in rows[:5]:
            print(f"  {r['filename']}  →  {r['plate']}")


if __name__ == "__main__":
    main()
