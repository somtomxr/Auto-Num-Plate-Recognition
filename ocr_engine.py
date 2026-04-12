"""
OCR engine optimised for Indian license plates.

Pipeline:
  1. Upscale small crops to a minimum width for OCR accuracy
  2. Multi-pass preprocessing (CLAHE, bilateral filter, sharpening,
     Otsu threshold, adaptive threshold)
  3. Two-line plate detection via horizontal projection analysis
     (handles HSRP / embossed split plates)
  4. EasyOCR as primary engine
  5. Tesseract (pytesseract) as a secondary engine
  6. Candidate scoring: prefer valid Indian format > confidence > length
"""

from __future__ import annotations

import re
import cv2
import numpy as np
from typing import Optional

from plate_utils import clean_plate, is_valid_indian_plate

_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ── Preprocessing ──────────────────────────────────────────────────────────────

def _upscale(img: np.ndarray, min_w: int = 300) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= min_w:
        return img
    scale = min_w / w
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def preprocess_plate(bgr: np.ndarray) -> list[np.ndarray]:
    """
    Return several preprocessed grayscale variants of the plate crop.

        Returns (in order):
            [0] clahe_sharpened  – best general-purpose image
            [1] otsu_white       – white text on dark background
            [2] otsu_dark        – dark text on white background (inverted)
            [3] adaptive         – handles uneven illumination plates
            [4] morph_close      – fills thin broken character strokes
    """
    bgr = _upscale(bgr, 300)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Bilateral filter: reduces noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Unsharp mask sharpening
    blur = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)

    # Otsu: white text on dark
    _, otsu_w = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu: dark text on white
    _, otsu_d = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive threshold (handles gradient/shadow backgrounds)
    adaptive = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 3,
    )

    # Morphological close helps reconnect fragmented strokes in embossed plates.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_close = cv2.morphologyEx(otsu_w, cv2.MORPH_CLOSE, k, iterations=1)

    return [sharpened, otsu_w, otsu_d, adaptive, morph_close]


# ── Two-line split detection ────────────────────────────────────────────────────

def find_split_line(gray: np.ndarray) -> Optional[int]:
    """
    Detect whether a plate has two lines of text (HSRP / embossed style).

    Indian two-line HSRP plates have:
      Line 1: State code + district  (e.g. MH 12)
      Line 2: Series + number        (e.g. AB 1234)

    Uses horizontal projection: a clear valley between the two text bands
    indicates a split.

    Returns the y-coordinate of the split, or None if single-line plate.
    """
    h, w = gray.shape
    if h < 40 or w < 60:
        return None  # too small

    # Binarise with inverted Otsu so text pixels are bright
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection: mean pixel intensity per row
    row_density = np.sum(binary, axis=1).astype(float) / (w + 1e-6)

    # Smooth to reduce single-pixel noise
    kernel = np.ones(3) / 3
    row_density = np.convolve(row_density, kernel, mode="same")

    # Search only the middle 40 % of plate height to avoid margins
    s = int(h * 0.30)
    e = int(h * 0.70)
    if e <= s:
        return None

    mid = row_density[s:e]
    min_idx = int(np.argmin(mid))
    split_y = min_idx + s

    # The gap must be significantly below the average text density
    text_rows = row_density[row_density > 5]
    if len(text_rows) == 0:
        return None
    avg_density = float(np.mean(text_rows))
    gap_density = float(row_density[split_y])

    if gap_density > avg_density * 0.30:
        return None  # no clear gap

    # Both halves must contain some text
    top_sum = float(np.sum(row_density[:split_y]))
    bot_sum = float(np.sum(row_density[split_y:]))
    if top_sum < 1 or bot_sum < 1:
        return None
    if min(top_sum, bot_sum) / max(top_sum, bot_sum) < 0.15:
        return None  # one half is nearly empty

    return split_y


# ── OCR helpers ─────────────────────────────────────────────────────────────────

def _easyocr_read(gray: np.ndarray, reader) -> tuple[str, float]:
    """Run EasyOCR on a grayscale image. Returns (joined_text, mean_confidence)."""
    try:
        results = reader.readtext(
            gray,
            detail=1,
            paragraph=False,
            allowlist=_ALLOWLIST,
            batch_size=1,
            decoder="greedy",
            text_threshold=0.40,
            low_text=0.25,
            link_threshold=0.20,
            width_ths=0.70,
            height_ths=0.40,
            adjust_contrast=0,   # we do our own contrast adjustment
        )
    except Exception:
        return "", 0.0

    if not results:
        return "", 0.0

    tokens = []
    confs = []
    for r in results:
        box, txt, conf = r
        if not txt:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        h = float(max(ys) - min(ys))
        tokens.append((cy, cx, txt, h))
        confs.append(float(conf))

    if not tokens:
        return "", 0.0

    # Sort roughly by text line then by left-to-right position.
    med_h = float(np.median([t[3] for t in tokens]))
    line_tol = max(6.0, med_h * 0.7)
    tokens.sort(key=lambda t: (round(t[0] / line_tol), t[1]))

    merged = "".join(t[2] for t in tokens)
    return merged, float(np.mean(confs))


def _parse_tess_conf(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return -1.0
    return v


def _tesseract_read(gray: np.ndarray) -> tuple[str, float]:
    """
    Run Tesseract with PSM 7 (single text line) and PSM 8 (single word).
    Returns (best_text, confidence).
    """
    try:
        import pytesseract

        _CFG = (
            f"--oem 3 "
            f"-c tessedit_char_whitelist={_ALLOWLIST} "
        )
        best_text, best_conf = "", 0.0
        for psm in (6, 7, 8):
            cfg = f"{_CFG} --psm {psm}"
            data = pytesseract.image_to_data(
                gray,
                config=cfg,
                output_type=pytesseract.Output.DICT,
            )
            words = []
            confs = []
            for t, c in zip(data.get("text", []), data.get("conf", [])):
                if not str(t).strip():
                    continue
                c_val = _parse_tess_conf(c)
                if c_val <= 20:
                    continue
                words.append(str(t))
                confs.append(c_val / 100.0)
            if words:
                joined = "".join(words)
                avg_conf = float(np.mean(confs)) if confs else 0.0
                if avg_conf > best_conf:
                    best_text, best_conf = joined, avg_conf

        return best_text, best_conf

    except Exception:
        return "", 0.0


# ── Two-line OCR combiner ───────────────────────────────────────────────────────

def _ocr_two_line(gray: np.ndarray, split_y: int, reader) -> tuple[str, float]:
    """
    OCR each half of a two-line plate separately then combine.

    Top line typically contains: STATE CODE + DISTRICT  (e.g. "MH12")
    Bottom line typically contains: SERIES + NUMBER     (e.g. "AB1234")

    A small vertical padding is added between the split and the crop
    to avoid cutting off descenders/ascenders.
    """
    h = gray.shape[0]
    pad = max(2, h // 20)

    top_crop = gray[: max(1, split_y - pad), :]
    bot_crop = gray[min(h - 1, split_y + pad) :, :]

    # Upscale halves individually (they're thin)
    top_crop = _upscale(top_crop, 280)
    bot_crop = _upscale(bot_crop, 280)

    top_txt, top_conf = _easyocr_read(top_crop, reader)
    bot_txt, bot_conf = _easyocr_read(bot_crop, reader)

    if not top_txt and not bot_txt:
        return "", 0.0

    combined = top_txt.strip() + bot_txt.strip()
    conf = (top_conf + bot_conf) / 2.0
    return combined, conf


# ── Candidate scoring ───────────────────────────────────────────────────────────

def _score(raw: str, cleaned: str, conf: float) -> float:
    """
    Score a candidate reading.

    Criteria (weighted sum):
      • Valid Indian plate format → large bonus
      • Valid state code prefix   → medium bonus
      • Reasonable length (7-10)  → small bonus
      • OCR confidence            → direct contribution
    """
    from plate_utils import STATE_CODES

    valid_bonus = 3.0 if is_valid_indian_plate(cleaned) else 0.0
    state_bonus = 1.0 if (len(cleaned) >= 2 and cleaned[:2] in STATE_CODES) else 0.0

    # Prefer modern plate lengths (10 chars standard, 11 with 3-letter series).
    n = len(cleaned)
    if n in (10, 11):
        length_bonus = 0.9
    elif n == 9:
        length_bonus = 0.25
    else:
        length_bonus = 0.1

    tail4_bonus = 0.4 if n >= 4 and cleaned[-4:].isdigit() else 0.0
    tail3_bonus = 0.1 if n >= 3 and cleaned[-3:].isdigit() else 0.0

    # Penalize likely truncation when state+RTO prefix is present but total length is short.
    trunc_penalty = -0.35 if (n == 9 and state_bonus > 0 and cleaned[-3:].isdigit()) else 0.0

    return valid_bonus + state_bonus + conf + length_bonus + tail4_bonus + tail3_bonus + trunc_penalty


def _prefer_complete_candidate(
    scored: list[tuple[float, str, str, float]],
    best_idx: int,
) -> int:
    """Prefer a complete candidate when best one looks like a 1-char truncation."""
    best_score, _, best_cleaned, _ = scored[best_idx]
    n = len(best_cleaned)

    # Target common miss: 9-char read where true plate is 10-char with one more trailing digit.
    if n != 9:
        return best_idx

    replacement_idx = best_idx
    replacement_score = best_score

    for i, (score, _raw, cleaned, _conf) in enumerate(scored):
        if len(cleaned) != 10:
            continue
        if not cleaned.endswith(tuple("0123456789")):
            continue
        if cleaned[:-1] != best_cleaned:
            continue

        # If a complete candidate is close in score, prefer it.
        if score >= best_score - 0.30 and score > replacement_score - 1e-9:
            replacement_idx = i
            replacement_score = score

    return replacement_idx


def _merge_candidates(candidates: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    """Merge repeated cleaned reads and boost confidence via consensus."""
    merged: dict[str, tuple[str, float, int]] = {}
    for raw, cleaned, conf in candidates:
        key = cleaned.strip()
        if not key:
            continue
        if key in merged:
            prev_raw, prev_conf, cnt = merged[key]
            merged[key] = (prev_raw if prev_conf >= conf else raw, max(prev_conf, conf), cnt + 1)
        else:
            merged[key] = (raw, conf, 1)

    out = []
    for cleaned, (raw, best_conf, cnt) in merged.items():
        boosted = min(1.0, best_conf + 0.05 * (cnt - 1))
        out.append((raw, cleaned, boosted))
    return out


def _expand_candidate(raw: str, conf: float) -> list[tuple[str, str, float]]:
    """
    Generate extra candidate variants from a noisy OCR string.

    OCR often includes one extra leading/trailing character from plate borders.
    This function trims small edges and re-cleans each variant.
    """
    src = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if not src:
        return []

    variants: list[tuple[str, str, float]] = []
    seen: set[str] = set()

    for ltrim in range(0, 3):
        for rtrim in range(0, 3):
            if ltrim + rtrim >= len(src):
                continue
            v = src[ltrim: len(src) - rtrim if rtrim else len(src)]
            if len(v) < 7:
                continue
            cleaned = clean_plate(v)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            penalty = 0.03 * (ltrim + rtrim)
            variants.append((v, cleaned, max(0.0, conf - penalty)))

    return variants


def _expand_missing_tail_zero(
    candidates: list[tuple[str, str, float]],
) -> list[tuple[str, str, float]]:
    """
    Recover cases where OCR drops the final trailing digit, commonly '0'.

    For plausible 9-char modern plates (AA##X####), add AA##X####0 as a candidate.
    """
    out = list(candidates)
    seen = {c[1] for c in candidates}

    for raw, cleaned, conf in candidates:
        if len(cleaned) != 9:
            continue
        if len(cleaned) < 6:
            continue
        if not (cleaned[:2].isalpha() and cleaned[2:4].isdigit()):
            continue
        if not cleaned[-3:].isdigit():
            continue

        z = cleaned + "0"
        if z in seen:
            continue

        seen.add(z)
        out.append((raw + "0", z, max(0.0, conf - 0.05)))

    return out


# ── Main entry point ────────────────────────────────────────────────────────────

def read_plate(
    crop_bgr: np.ndarray,
    reader,
    *,
    use_tesseract: bool = True,
) -> dict:
    """
    Full OCR pipeline for a YOLO-detected plate crop.

    Parameters
    ----------
    crop_bgr : np.ndarray
        BGR image of the plate region (raw crop from YOLO).
    reader : easyocr.Reader
        Pre-loaded EasyOCR reader instance.
    use_tesseract : bool
        Whether to also try Tesseract as a secondary engine.

    Returns
    -------
    dict with keys:
        text        – str   : cleaned, corrected plate text
        confidence  – float : best OCR confidence (0-1)
        two_line    – bool  : True if a 2-line layout was detected
        raw         – str   : raw text before cleaning (for debugging)
    """
    # 1. Preprocess ──────────────────────────────────────────────────────────────
    imgs = preprocess_plate(crop_bgr)
    primary_gray = imgs[0]   # CLAHE-sharpened (best general reference)

    # 2. Two-line detection ──────────────────────────────────────────────────────
    split_y  = find_split_line(primary_gray)
    two_line = split_y is not None

    # 3. Collect OCR candidates ──────────────────────────────────────────────────
    # Each candidate: (raw_text, cleaned_text, confidence)
    candidates: list[tuple[str, str, float]] = []

    if two_line:
        raw2, conf2 = _ocr_two_line(primary_gray, split_y, reader)
        if raw2.strip():
            cleaned2 = clean_plate(raw2)
            candidates.append((raw2, cleaned2, conf2))
            candidates.extend(_expand_candidate(raw2, conf2))

    # Multi-pass full-plate OCR across all preprocessed images
    for img_variant in imgs:
        raw, conf = _easyocr_read(img_variant, reader)
        if raw.strip():
            candidates.append((raw, clean_plate(raw), conf))
            candidates.extend(_expand_candidate(raw, conf))

    # Tesseract passes (sharpened + Otsu only, for speed)
    if use_tesseract:
        for img_variant in imgs[:2]:
            raw, conf = _tesseract_read(img_variant)
            if raw.strip():
                candidates.append((raw, clean_plate(raw), conf))
                candidates.extend(_expand_candidate(raw, conf))

    # 4. Fallback: run all passes again without allowlist restriction ─────────────
    if not candidates:
        for img_variant in imgs:
            raw, conf = _easyocr_read(img_variant, reader)
            if raw.strip():
                candidates.append((raw, clean_plate(raw), conf))
                candidates.extend(_expand_candidate(raw, conf))

    candidates = _expand_missing_tail_zero(candidates)
    candidates = _merge_candidates(candidates)

    if not candidates:
        return {"text": "", "confidence": 0.0, "two_line": two_line, "raw": ""}

    # 5. Select best candidate ───────────────────────────────────────────────────
    scored = [(_score(raw, cleaned, conf), raw, cleaned, conf) for raw, cleaned, conf in candidates]
    best_idx = max(range(len(scored)), key=lambda i: scored[i][0])
    best_idx = _prefer_complete_candidate(scored, best_idx)
    _, best_raw, best_cleaned, best_conf = scored[best_idx]

    return {
        "text":       best_cleaned,
        "confidence": round(best_conf, 4),
        "two_line":   two_line,
        "raw":        best_raw,
    }
