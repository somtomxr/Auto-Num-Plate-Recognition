"""
Indian license plate utilities: validation, cleaning, formatting.

Supports:
  - Standard plates    : MH12AB1234
  - Fancy / HSRP       : MH 12 AB 1234  (spaced variant)
  - BH-series plates   : 22BH1234AA
  - Old single-letter  : MH12A1234
  - Two-line (HSRP)    : handled by ocr_engine
"""

from __future__ import annotations

import re
from typing import Optional

# ── Character correction maps ──────────────────────────────────────────────────
# Digits that are commonly misread as letters in positions that require letters
_DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G",
}

# Letters that are commonly misread as digits in positions that require digits
_LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "Q": "0",
    "L": "1",
    "D": "0",
    "U": "0",
    "T": "1",
}

# All valid Indian state / UT codes (ISO 3166-2:IN)
STATE_CODES: frozenset[str] = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TG", "TN", "TR", "TS", "UK", "UP", "WB",
})

# Plate format regexes. State-prefixed formats are checked together with
# STATE_CODES in is_valid_indian_plate().
_STATE_PLATE_PATTERNS: list[re.Pattern] = [
    # Standard modern: MH12AB1234
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$"),
    # Single-series:   MH12A1234
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]\d{4}$"),
    # Three-letter:    MH12ABC1234
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$"),
    # Old short / partial but plausible: MH12AB123
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,3}$"),
]

_BH_PLATE_PATTERN = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")


def _to_letter(ch: str) -> str:
    return _DIGIT_TO_LETTER.get(ch, ch)


def _to_digit(ch: str) -> str:
    return _LETTER_TO_DIGIT.get(ch, ch)


def _repair_state_prefix(plate: str) -> str:
    """Repair first two characters to the nearest valid Indian state code."""
    if len(plate) < 2:
        return plate

    prefix = "".join(_to_letter(c) for c in plate[:2])
    if prefix in STATE_CODES:
        return prefix + plate[2:]

    # If OCR is off by one character, snap to the closest known state code.
    nearest = [
        code for code in STATE_CODES
        if sum(1 for i in range(2) if code[i] != prefix[i]) <= 1
    ]
    if len(nearest) == 1:
        return nearest[0] + plate[2:]

    return prefix + plate[2:]


def clean_plate(raw: str) -> str:
    """
    Apply position-aware character correction for Indian plate format.

    Rules:
      pos 0-1  → must be letters  (state code, e.g. MH)
      pos 2-3  → must be digits   (RTO code,   e.g. 12)
      pos 4-6  → must be letters  (series,      e.g. AB)
      pos 7-10 → must be digits   (number,      e.g. 1234)

    BH-series has different layout and is handled separately.
    """
    if not raw:
        return ""

    # Normalise
    p = raw.upper().strip()
    p = re.sub(r"\bIND\b", "", p)        # strip embossed IND
    p = re.sub(r"[^A-Z0-9]", "", p)     # keep only alphanumerics
    p = p.strip()

    if len(p) < 4:
        return p

    chars = list(p)
    n = len(chars)

    # ── Pre-correction noise trims ──────────────────────────────────────────────
    # Drop a leading single digit when the next two chars are letters
    # (OCR picks up part of the number plate border/frame as a digit)
    if n > 9 and chars[0].isdigit() and chars[1].isalpha() and chars[2].isalpha():
        chars = chars[1:]
        n -= 1

    # Drop a trailing extra digit when the last 5 characters are all digits
    # (standard plates only have 4 trailing digits)
    if n > 10 and all(c.isdigit() for c in chars[-5:]):
        chars = chars[:-1]
        n -= 1

    # ── BH-series: starts with two digits ──────────────────────────────────────
    if chars[0].isdigit() and chars[1].isdigit():
        # pos 2-3: letters (BH)
        for i in range(2, min(4, n)):
            chars[i] = _to_letter(chars[i])
        # pos 4-7: digits
        for i in range(4, min(8, n)):
            chars[i] = _to_digit(chars[i])
        # pos 8+: letters (suffix)
        for i in range(8, n):
            chars[i] = _to_letter(chars[i])
        return "".join(chars)

    # ── Standard plate ─────────────────────────────────────────────────────────
    # Positions 0-1: state code letters
    for i in range(0, min(2, n)):
        chars[i] = _to_letter(chars[i])

    # Positions 2-3: RTO digits
    for i in range(2, min(4, n)):
        chars[i] = _to_digit(chars[i])

    # Positions 4+: series letters then 4 trailing digits
    # Standard Indian plate: AA ## [A-Z]{1,3} ####
    # Total length → 9 (1-series), 10 (2-series), 11 (3-series)
    # Reliable rule: the last 4 characters are always the registration number (digits)
    if n >= 9:
        series_end = n - 4   # exclusive index — last 4 are digits
        series_end = max(5, min(series_end, 7))  # clamp: series is 1–3 chars
        for i in range(4, series_end):
            chars[i] = _to_letter(chars[i])
        for i in range(series_end, n):
            chars[i] = _to_digit(chars[i])
    else:
        # Short / partial read — best-effort: treat pos 4-5 as letters, rest digits
        for i in range(4, min(6, n)):
            chars[i] = _to_letter(chars[i])
        for i in range(6, n):
            chars[i] = _to_digit(chars[i])

    result = _repair_state_prefix("".join(chars))
    return result


def is_valid_indian_plate(plate: str) -> bool:
    """Return True if plate matches a known Indian plate format."""
    cleaned = re.sub(r"[^A-Z0-9]", "", plate.upper())

    if _BH_PLATE_PATTERN.match(cleaned):
        return True

    if len(cleaned) < 2 or cleaned[:2] not in STATE_CODES:
        return False

    for pat in _STATE_PLATE_PATTERNS:
        if pat.match(cleaned):
            return True

    # Fallback structural check for partial but plausible reads.
    if (
        len(cleaned) >= 8
        and cleaned[:2] in STATE_CODES
        and cleaned[2:4].isdigit()
        and any(c.isalpha() for c in cleaned[4:])
        and any(c.isdigit() for c in cleaned[4:])
    ):
        return True

    return False


def format_plate(plate: str) -> str:
    """Return a human-readable formatted plate string, e.g. 'MH 12 AB 1234'."""
    p = re.sub(r"[^A-Z0-9]", "", plate.upper())
    if _BH_PLATE_PATTERN.match(p):
        # 22 BH 1234 AA
        return f"{p[:2]} {p[2:4]} {p[4:8]} {p[8:]}"
    if len(p) == 11:
        # MH 12 ABC 1234
        return f"{p[:2]} {p[2:4]} {p[4:7]} {p[7:]}"
    if len(p) == 10:
        # MH 12 AB 1234
        return f"{p[:2]} {p[2:4]} {p[4:6]} {p[6:]}"
    if len(p) == 9:
        # MH 12 A 1234
        return f"{p[:2]} {p[2:4]} {p[4:5]} {p[5:]}"
    return p
