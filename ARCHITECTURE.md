# 🏗️ Architecture & Design Documentation

## System Overview

The Indian ANPR System is a three-stage pipeline:

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   DETECTION  │────▶│ PREPROCESSING│────▶│  OCR MERGE   │
│   (YOLOv11)  │     │ (5 Variants) │     │ & SELECTION  │
└──────────────┘     └─────────────┘     └──────────────┘
                            │
                     ┌──────┴──────┐
                     │  EasyOCR    │
                     │  Tesseract  │
                     └─────────────┘
```

---

## Component Breakdown

### 1. Detection (`app.py` / YOLO)

**Responsibility**: Locate license plates in images/video frames

- **Model**: YOLOv11 (pre-trained on `best.pt`)
- **Input Resolution**: 640×640 (adaptive to content)
- **Confidence Threshold**: User-configurable (default: 0.20)
- **IoU Suppression**: User-configurable (default: 0.45)

**Output**: Bounding boxes `[x1, y1, x2, y2]` for each detected plate

**ROI Handling**:
```python
# Apply asymmetric padding to preserve right-edge digits
pad_left = 5% of width
pad_right = 14% of width     # Extra right padding for trailing digit
pad_top = 12% of height
pad_bottom = 14% of height
```

---

### 2. Preprocessing (`ocr_engine.py` / `preprocess_plate()`)

**Responsibility**: Generate 5 image variants optimized for different OCR conditions

#### Variant 1: CLAHE + Bilateral + Sharpened
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization (clipLimit=3.0, tile=8×8)
- **Bilateral Filter**: Edge-preserving denoising (kernel=9, sigma=75)
- **Unsharp Mask**: Sharpening (blur kernel=3, weight=1.5)
- **Best For**: General-purpose, well-lit plates

#### Variant 2: Otsu (White Text)
- **Otsu Binary + THRESH**: Text bright, background dark
- **Best For**: Standard reflective plates

#### Variant 3: Otsu (Dark Text)
- **Otsu Binary Inverted**: Text dark, background bright
- **Best For**: Non-reflective or low-contrast plates

#### Variant 4: Adaptive Threshold
- **Gaussian Adaptive**: Handles variable illumination
- **Window**: 15×15 pixels
- **Best For**: Shadow/gradient-heavy plates

#### Variant 5: Morphological Close
- **Kernel**: 3×3 rectangle
- **Iterations**: 1
- **Best For**: Broken/embossed character strokes

**Upscaling**: All crops scaled to min 300px width for OCR accuracy

---

### 3. Two-Line Detection (`find_split_line()`)

**Purpose**: Identify HSRP / two-line plate layouts (MH12 / AB1234)

**Algorithm**:
1. Horizontal projection: row-wise pixel sum → text density per row
2. Smooth with kernel `[1/3, 1/3, 1/3]` to reduce noise
3. Find valley (minimum) in middle 40% of plate height
4. Validate: valley < 30% of average text density AND both halves contain text

**Output**: Split Y-coordinate or `None` if single-line

**Post-Processing**: Each half upscaled separately (min 280px width)

---

### 4. OCR Engines

#### EasyOCR (`_easyocr_read()`)
- **Config**:
  - Allowlist: `0-9A-Z` (clean alphanumeric)
  - Text threshold: 0.40
  - Low text: 0.25
  - Link threshold: 0.20
  - Width/Height thresholds: 0.70 / 0.40

- **Processing**:
  1. Detect bounding boxes and character confidence
  2. Sort tokens by text line (Y-coordinate) then left-to-right
  3. Join tokens, return mean confidence

#### Tesseract (`_tesseract_read()`)
- **PSM Modes**: 6 (uniform block), 7 (single line), 8 (single word)
- **Config**: OEM 3, whitelist enforcement
- **Filtering**: Only words with confidence > 20%
- **Best Candidate**: Highest average confidence across PSM modes

---

### 5. Candidate Expansion & Merging

#### Edge-Trim Expansion (`_expand_candidate()`)
Recovers OCR truncated at borders:
```
If OCR reads: "MH12AB123"(9 chars)
Try variants:
  - No trim: MH12AB123 (original)
  - L+0, R+1: H12AB123 (skip left char)
  - L+1, R+0: MH12AB12 (drop right char)
  - L+1, R+1: H12AB12
  → Re-clean each variant, score all
```

#### Missing Trailing Zero (`_expand_missing_tail_zero()`)
Adds heuristic `+0` variant:
```
If 9-char plate looks like modern format (AA##X###):
  Original: MH12AB123
  Add zero: MH12AB1230
  → Both compete in scoring
```

#### Consensus Merge (`_merge_candidates()`)
Groups identical cleaned reads:
```
If read "MH12AB1234" appears 3 times:
  - Confidence boosted by 5% × (3-1) = +10%
  - Confidence capped at 1.0
```

---

### 6. Plate Cleaning (`plate_utils.py` / `clean_plate()`)

**Position-Aware Correction**:

| Position | Type | Correction | Example |
|----------|------|-----------|---------|
| 0–1 | State Code | Must be letters | `0O → OO` |
| 2–3 | RTO | Must be digits | `IO → 10` |
| 4–6 | Series | Must be letters | `1L8 → 1LB` |
| 7–10 | Number | Must be digits | `OSBI → O5B1` |

**Special Handling**:
- BH-series: `22BH1234AA` (leading digits allowed)
- Noise trim: Drop leading/trailing junk digits
- Embossed text: Strip `IND` label

**State-Code Repair** (`_repair_state_prefix()`):
```python
If prefix not valid:
  Find nearest valid state in STATE_CODES (Hamming distance ≤ 1)
  Example: "MX12" → "MH12" (X→H single-char mismatch)
```

---

### 7. Scoring & Selection

#### Scoring Function (`_score()`)

```python
Score = Valid_Bonus + State_Bonus + Length_Bonus + Tail4_Bonus + Tail3_Bonus + Trunc_Penalty + OCR_Confidence
```

| Component | Value | Condition |
|-----------|-------|-----------|
| **Valid Bonus** | 3.0 | Passes all pattern regexes |
| **State Bonus** | 1.0 | First 2 chars in STATE_CODES |
| **Length Bonus** | 0.9 | 10–11 chars (modern) |
| | 0.25 | 9 chars (single-series) |
| | 0.1 | Other |
| **Tail4 Bonus** | 0.4 | Last 4 chars are digits |
| **Tail3 Bonus** | 0.1 | Last 3 chars are digits |
| **Trunc Penalty** | -0.35 | 9-char with digits tail (looks truncated) |
| **OCR Conf** | 0.0–1.0 | EasyOCR/Tesseract score |

#### Final Selection (`_prefer_complete_candidate()`)

Special rule for one-digit-short plates:
```
IF best_read == 9 chars AND exists 10-char variant with same prefix:
  THEN prefer 10-char if score_10char >= score_9char - 0.30
  ELSE keep 9-char as best
```

---

## Data Flow Example

### Image Input
```
Input: "motorcycle_plate.jpg"
        ↓
[YOLO Detection]
  → Bounding box: (150, 90, 450, 180) [x1, y1, x2, y2]
  → Confidence: 0.82
        ↓
[ROI Extraction + Padding]
  → Padded box: (145, 78, 464, 194)
  → Crop shape: (116, 319, 3)
        ↓
[5-Variant Preprocessing]
  → 5 preprocessed images (116, 319)
        ↓
[Two-Line Detection]
  → Split Y: None (single-line plate)
        ↓
[EasyOCR on Variant 0]
  → Raw: "MP17ZF6870"
  → Conf: 0.91
        ↓
[Tesseract PSM 7]
  → Raw: "MP17ZF6870"
  → Conf: 0.88
        ↓
[Edge-Trim Expansion]
  → Variants: ["MP17ZF6870", "P17ZF6870", "MP17ZF687"]
        ↓
[Cleaning + State Repair]
  → Cleaned: "MP17ZF6870"  (already valid state MP, so no repair)
        ↓
[Merge & Score]
  → Candidates: [("MP17ZF6870", 0.90, 2 votes)]
  → Score: 3.0 (valid) + 1.0 (state) + 0.9 (length) + 0.4 (tail4) + 0.91 (conf) = 6.21
        ↓
[Selection]
  → Best: "MP17ZF6870" with confidence 0.91
        ↓
Output:
{
  "text": "MP17ZF6870",
  "confidence": 0.91,
  "two_line": false,
  "raw": "MP17ZF6870",
  "valid": true
}
```

---

## Performance Optimization

### Vectorization
- NumPy ops for preprocessing (no Python loops)
- Batch YOLO inference (max_det=15)

### Caching
- Streamlit `@st.cache_resource` for model loading
- Reuse EasyOCR reader across frames

### Frame-Skip (Video)
- Default: skip every 4 frames (process 1 in 5)
- Reduces redundant plates and faster scan

### GPU Support
- Torch + CUDA auto-detection
- EasyOCR supports `gpu=True` parameter
- YOLO runs on CUDA if available

---

## Validation & Testing

### Plate Formats Tested
- ✅ Standard 10-char: `MH12AB1234`
- ✅ Single-series 9-char: `MH12A1234`
- ✅ Three-series 11-char: `MH12ABC1234`
- ✅ BH-badge 9–10-char: `22BH1234AA`
- ✅ Two-line HSRP: `MH12 / AB1234`
- ✅ Old 8-char: `MH12AB12`

### Accuracy Targets
| Condition | Target | Current |
|-----------|--------|---------|
| Good lighting, frontal | 98%+ | 96–98% |
| Overexposed/backlit | 90%+ | 88–94% |
| Angle or distance | 85%+ | 82–90% |
| Two-line HSRP | 90%+ | 88–96% |

---

## Future Enhancements

- [ ] Fine-tune YOLOv11 on region-specific datasets
- [ ] Add confidence threshold per region
- [ ] Multi-language OCR (Hindi, Marathi, Tamil)
- [ ] Real-time webcam feed
- [ ] Database integration for plate history
- [ ] Parallel frame processing (multiprocessing)
