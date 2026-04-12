# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] – 2026-04-12

### Added
- ✨ Trailing zero recovery heuristic for 10-digit modern plates
- ✨ Missing digit reconstruction via edge-trim candidate expansion
- ✨ Wider right-margin ROI padding to preserve final digits
- ✨ State code prefix auto-repair (nearest neighbor snap-to-valid)
- ✨ Consensus-based candidate merging (repeated reads boost score)
- ✨ Modern plate length preference in scoring (10/11 > 9-char)
- ✨ Full architecture & contribution documentation
- ✨ Comprehensive README with API examples and troubleshooting

### Improved
- 🔧 OCR token sorting before merge (line-aware, left-to-right)
- 🔧 Robust Tesseract confidence parsing (handle float conversion)
- 🔧 EasyOCR preprocessing with morphological close variant
- 🔧 Stricter fallback validity validation
- 🔧 Added PSM 6 to Tesseract multi-pass (block detection mode)
- 🔧 UI polish: gradient backgrounds, enhanced card styling, button improvements

### Fixed
- 🐛 TypeError in `_plate_html()` due to unexpected keyword arguments
- 🐛 OCR detection crashes on extra dictionary fields
- 🐛 Last digit (typically 0) clipping on modern 10-char plates
- 🐛 Repeated OCR misreads not being differentiated in merging
- 🐛 Loose fallback validation accepting invalid state codes
- 🐛 Browser frontend cache preventing updated code from loading

### Changed
- 📝 Refactored OCR candidate generation (now uses `_expand_missing_tail_zero`)
- 📁 Cleaned up requirements.txt (removed duplicates)
- 📁 Expanded .gitignore with build artifacts and environment files

---

## [1.1.0] – 2026-04-11

### Added
- ✨ Video processing with frame-skipping & batch detection
- ✨ Multi-engine OCR fallback (EasyOCR + Tesseract primary + secondary)
- ✨ Two-line HSRP plate auto-detection & separate OCR per half
- ✨ Position-aware character correction for Indian plate formats
- ✨ Detection log with CSV export functionality
- ✨ Statistics dashboard (total, valid, 2-line, avg confidence)
- ✨ Settings UI for confidence, IoU, strict format, Tesseract toggle

### Improved
- 🔧 YOLO detection with adaptive imgsz and max_det tuning
- 🔧 5-variant preprocessing pipeline (CLAHE, Otsu, Adaptive, Morphology)
- 🔧 Enhanced crop padding for better edge preservation
- 🔧 BH-series plate format support
- 🔧 Dark-mode dark theme with GitHub colors

---

## [1.0.0] – 2026-04-01

### Added
- ✨ Initial release with YOLOv11 plate detection
- ✨ EasyOCR integration for character recognition
- ✨ Basic plate validation for standard Indian formats
- ✨ Streamlit web UI with image upload
- ✨ Model weights (best.pt) trained on Indian plates

### Features
- 📷 Single image detection and display
- 🎯 Bounding box visualization
- 📊 Basic confidence scoring
- 🖼️ Dark-mode UI with custom CSS

---

## [Unreleased]

### Planned
- [ ] GPU batch inference optimization
- [ ] Real-time webcam feed support
- [ ] Multi-language OCR (Hindi, Marathi, Tamil)
- [ ] Database backend for plate history
- [ ] REST API wrapper
- [ ] Mobile app (React Native / Flutter)
- [ ] Docker containerization
- [ ] Fine-tuned YOLOv11 models per region
- [ ] Performance benchmarking suite
- [ ] Plate anonymization for privacy

---

## How to Update

```bash
# Check current version
python -c "import app; print(app.__version__)" 2>/dev/null || echo "No version file"

# Pull latest changes
git pull origin main

# Install new dependencies
pip install --upgrade -r requirements.txt

# Restart app
streamlit run app.py
```

---

**📌 Note**: Versions prior to 1.0.0 are development/pre-releases.
