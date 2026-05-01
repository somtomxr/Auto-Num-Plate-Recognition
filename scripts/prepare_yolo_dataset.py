"""Prepare a YOLO dataset for Indian number plate detection.

Input:
    A raw dataset folder containing images and either:
    - Pascal VOC XML annotations, or
    - YOLO .txt annotations next to images / in label folders.

Output:
    datasets/plate_yolo/
      images/train, images/val
      labels/train, labels/val
      dataset.yaml

This trains one class only: plate.
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def image_size(image_path: Path) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Install requirements.txt first.") from exc

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def find_matching_xml(image_path: Path, raw_dir: Path) -> Path | None:
    candidates = [
        image_path.with_suffix(".xml"),
        image_path.parent.parent / "annotations" / f"{image_path.stem}.xml",
        image_path.parent.parent / "Annotations" / f"{image_path.stem}.xml",
        raw_dir / "annotations" / f"{image_path.stem}.xml",
        raw_dir / "Annotations" / f"{image_path.stem}.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(raw_dir.rglob(f"{image_path.stem}.xml"))
    return matches[0] if matches else None


def find_matching_yolo_txt(image_path: Path, raw_dir: Path) -> Path | None:
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent.parent / "Labels" / f"{image_path.stem}.txt",
        raw_dir / "labels" / f"{image_path.stem}.txt",
        raw_dir / "Labels" / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(raw_dir.rglob(f"{image_path.stem}.txt"))
    return matches[0] if matches else None


def voc_xml_to_yolo(xml_path: Path, image_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        width = int(float(size.findtext("width", "0")))
        height = int(float(size.findtext("height", "0")))
    else:
        width, height = image_size(image_path)

    if width <= 0 or height <= 0:
        width, height = image_size(image_path)

    rows: list[str] = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))

        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        rows.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    return rows


def normalize_existing_yolo(txt_path: Path) -> list[str]:
    rows: list[str] = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            _, x, y, w, h = parts
            values = [float(x), float(y), float(w), float(h)]
        except ValueError:
            continue
        if all(0.0 <= value <= 1.0 for value in values):
            rows.append(f"0 {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}")
    return rows


def annotation_rows(image_path: Path, raw_dir: Path) -> list[str]:
    xml_path = find_matching_xml(image_path, raw_dir)
    if xml_path:
        return voc_xml_to_yolo(xml_path, image_path)

    txt_path = find_matching_yolo_txt(image_path, raw_dir)
    if txt_path:
        return normalize_existing_yolo(txt_path)

    return []


def reset_output(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_sample(
    image_path: Path,
    rows: list[str],
    out_dir: Path,
    split: str,
    index: int,
) -> None:
    safe_stem = f"{index:06d}_{image_path.stem}"
    image_out = out_dir / "images" / split / f"{safe_stem}{image_path.suffix.lower()}"
    label_out = out_dir / "labels" / split / f"{safe_stem}.txt"

    shutil.copy2(image_path, image_out)
    label_out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_yaml(out_dir: Path) -> None:
    yaml_text = f"""path: {out_dir.resolve()}
train: images/train
val: images/val
names:
  0: plate
"""
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw plate dataset to YOLO format.")
    parser.add_argument("raw_dir", type=Path, help="Extracted Kaggle dataset folder.")
    parser.add_argument("--out", type=Path, default=Path("datasets/plate_yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise SystemExit(f"Raw dataset folder not found: {args.raw_dir}")

    images = find_images(args.raw_dir)
    samples: list[tuple[Path, list[str]]] = []
    skipped = 0

    for image_path in images:
        rows = annotation_rows(image_path, args.raw_dir)
        if rows:
            samples.append((image_path, rows))
        else:
            skipped += 1

    if not samples:
        raise SystemExit(
            "No annotated images found. Expected Pascal VOC XML or YOLO txt annotations."
        )

    random.seed(args.seed)
    random.shuffle(samples)

    reset_output(args.out)

    val_count = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    for idx, (image_path, rows) in enumerate(train_samples):
        copy_sample(image_path, rows, args.out, "train", idx)
    for idx, (image_path, rows) in enumerate(val_samples):
        copy_sample(image_path, rows, args.out, "val", idx)

    write_yaml(args.out)

    print("YOLO dataset prepared")
    print("=====================")
    print(f"Raw images found: {len(images)}")
    print(f"Annotated samples used: {len(samples)}")
    print(f"Skipped without annotations: {skipped}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Dataset YAML: {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
