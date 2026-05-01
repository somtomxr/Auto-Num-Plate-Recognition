"""Validate and summarize manually collected ANPR failure samples."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ALLOWED_PROBLEM_TYPES = {"no_detection", "wrong_ocr", "works"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ANPR failure samples.")
    parser.add_argument("labels_csv", type=Path, help="CSV with filename,actual_plate,problem_type,notes")
    args = parser.parse_args()

    if not args.labels_csv.exists():
        raise SystemExit(f"CSV not found: {args.labels_csv}")

    base_dir = args.labels_csv.parent
    counts: Counter[str] = Counter()
    missing_files: list[str] = []
    invalid_rows: list[str] = []
    total = 0

    with args.labels_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"filename", "actual_plate", "problem_type"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise SystemExit(f"Missing columns: {', '.join(sorted(missing_columns))}")

        for line_no, row in enumerate(reader, start=2):
            total += 1
            filename = (row.get("filename") or "").strip()
            actual_plate = (row.get("actual_plate") or "").strip()
            problem_type = (row.get("problem_type") or "").strip()

            if not filename or not actual_plate or problem_type not in ALLOWED_PROBLEM_TYPES:
                invalid_rows.append(str(line_no))
                continue

            counts[problem_type] += 1
            if not (base_dir / filename).exists():
                missing_files.append(filename)

    print("Failure sample summary")
    print("======================")
    print(f"Total rows: {total}")
    for problem_type in ("no_detection", "wrong_ocr", "works"):
        print(f"{problem_type}: {counts[problem_type]}")

    if invalid_rows:
        print(f"\nRows to fix: {', '.join(invalid_rows)}")

    if missing_files:
        print("\nMissing image files:")
        for filename in missing_files:
            print(f"- {filename}")

    if not invalid_rows and not missing_files:
        print("\nCSV looks ready. Next: run the ANPR benchmark or ask Codex to analyze these samples.")


if __name__ == "__main__":
    main()
