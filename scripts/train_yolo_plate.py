"""Train a YOLO detector for one class: number plate."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO for plate detection.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Starting YOLO model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="plate_train")
    parser.add_argument("--device", default=None, help="Use 'cpu', '0' for GPU, or leave empty.")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"dataset.yaml not found: {args.data}")

    model = YOLO(args.model)
    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        train_args["device"] = args.device

    model.train(**train_args)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print("\nTraining complete")
    print("=================")
    print(f"Best model: {best}")
    print("To use it in the app:")
    print("  cp best.pt best_old.pt")
    print(f"  cp {best} best.pt")


if __name__ == "__main__":
    main()
