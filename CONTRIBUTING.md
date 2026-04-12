# 🤝 Contributing to Indian ANPR System

Thank you for your interest in contributing! This document outlines the process for submitting issues, feature requests, and pull requests.

---

## 📋 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on improving the project

---

## 🐛 Reporting Issues

### Before Submitting
1. Check existing issues to avoid duplicates
2. Collect diagnostic info:
   - OS and Python version
   - Plate image (anonymized if sensitive)
   - Settings used
   - Full error traceback

### Issue Template
```markdown
**Describe the problem**
[Clear description]

**Steps to reproduce**
1. Upload image...
2. Set confidence to...
3. Click Detect...

**Expected behavior**
Should detect: MH12AB1234

**Actual behavior**
Detected: MH12AB123 (missing last digit)

**Environment**
- OS: macOS 14.2
- Python: 3.11.2
- Streamlit: 1.28.0

**Additional context**
[Image, logs, etc.]
```

---

## 🚀 Feature Requests

### Suggest a Feature
1. Use the issue title: `[FEATURE] Brief description`
2. Explain the motivation
3. Provide use-case examples
4. Describe proposed implementation (if applicable)

### High-Impact Features
- Real-time webcam feed
- Database backend for plate history
- Multi-language OCR
- GPU batch optimization
- Mobile app integration

---

## 🔧 Development Setup

```bash
# Clone repo
git clone https://github.com/your-org/anpr-system.git
cd License-Plate-Recognition-app

# Create dev environment
python3 -m venv venv-dev
source venv-dev/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Verify setup
streamlit run app.py
```

---

## ✍️ Coding Standards

### Style Guide
- **Follow PEP 8**: Use `black` for formatting
- **Type hints**: Add type annotations to functions
- **Docstrings**: NumPy/Google style for public functions
- **Comments**: Explain "why", not "what"

### Example
```python
def clean_plate(raw: str) -> str:
    """
    Apply position-aware character correction.

    Position rules:
      0-1  → letters (state)
      2-3  → digits (RTO)
      4-6  → letters (series)
      7-10 → digits (number)

    Args:
        raw: Raw OCR text (e.g., "MH12AB1234")

    Returns:
        Cleaned plate text with corrected chars.
    """
    if not raw:
        return ""
    # ... implementation
```

### Pre-commit Checks
```bash
# Format code
black ocr_engine.py plate_utils.py app.py

# Lint
flake8 ocr_engine.py plate_utils.py app.py

# Type check
mypy ocr_engine.py plate_utils.py
```

---

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Write Tests
Place test files in `tests/` directory:

```python
# tests/test_plate_utils.py
import pytest
from plate_utils import clean_plate, is_valid_indian_plate

def test_clean_plate_standard():
    raw = "MH12AB1234"
    assert clean_plate(raw) == "MH12AB1234"

def test_clean_plate_with_noise():
    raw = "MH12AB1234X"  # Extra char
    assert clean_plate(raw) == "MH12AB1234"

def test_is_valid_indian_plate():
    assert is_valid_indian_plate("MH12AB1234")
    assert not is_valid_indian_plate("XX12AB1234")  # Invalid state
```

---

## 📝 Commit Messages

Follow **Conventional Commits**:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Test additions

### Examples
```
feat(ocr): add missing trailing-zero recovery

Previously, 10-digit plates would be truncated to 9 digits
when the final 0 was clipped during edge detection.

Add _expand_missing_tail_zero() heuristic to recover the
final digit for plates matching modern format (AA##X####).

Fixes #42
```

```
fix(preprocessing): increase right-margin padding

Trailing characters were clipped due to insufficient
right-side ROI padding.

Increase pad_right from bw*6% to bw*14%.
```

---

## 🔄 Pull Request Process

### Before Submitting
1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make changes and commit with proper messages
3. Update tests and documentation
4. Run linting and tests locally
5. Push to your fork

### PR Template
```markdown
## Description
[What does this PR do?]

## Related Issue
Closes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Manual testing on sample plates
- [ ] No regressions observed

## Checklist
- [ ] Code follows PEP 8
- [ ] Added/updated docstrings
- [ ] Updated README if needed
- [ ] Tested on macOS/Linux/Windows
```

### Review Process
1. Code review by maintainers
2. Address feedback and push updates
3. Approval → merge when CI passes

---

## 🎯 Priority Areas for Contribution

### High Priority
- [ ] GPU batch inference optimization
- [ ] Multi-language OCR support (Hindi, Marathi, Tamil)
- [ ] Real-time webcam input
- [ ] Database backend integration
- [ ] Fine-tune YOLOv11 on regional datasets

### Medium Priority
- [ ] Unit test coverage (→90%)
- [ ] Performance benchmarking suite
- [ ] Docker containerization
- [ ] REST API wrapper
- [ ] Plate anonymization for privacy

### Low Priority (Nice-to-Have)
- [ ] Web scraping for test datasets
- [ ] Visualization dashboard
- [ ] Export to multiple formats
- [ ] Batch processing CLI
- [ ] Slack/Telegram bot integration

---

## 📚 Documentation

### Update README
- Add feature descriptions
- Update performance metrics
- Include usage examples

### Add Architecture Docs
Keep `ARCHITECTURE.md` in sync with code changes.

### Docstrings
Use Google style:
```python
def read_plate(crop_bgr: np.ndarray, reader, *, use_tesseract: bool = True) -> dict:
    """Full OCR pipeline for a YOLO-detected plate crop.

    Long description here if needed.

    Args:
        crop_bgr: BGR image of the plate region.
        reader: Pre-loaded EasyOCR reader instance.
        use_tesseract: Whether to try Tesseract fallback.

    Returns:
        dict with keys:
            text (str): Cleaned, corrected plate text.
            confidence (float): Best OCR confidence (0-1).
            two_line (bool): True if 2-line layout detected.
            raw (str): Raw text before cleaning.

    Raises:
        ValueError: If crop_bgr has unexpected shape.
    """
```

---

## 🚢 Release Process

### Versioning
Follow **Semantic Versioning** (MAJOR.MINOR.PATCH):
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes

### Release Checklist
- [ ] Update version in code
- [ ] Update `requirements.txt` if needed
- [ ] Update `CHANGELOG.md`
- [ ] Create git tag: `git tag v1.2.3`
- [ ] Push tag: `git push origin v1.2.3`

---

## ❓ Questions?

- Open a **Discussion** issue
- Check **ARCHITECTURE.md** for design details
- Review existing issues for similar questions

---

**Thank you for contributing! 🎉**
