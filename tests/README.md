# Tests Folder

Tests prove that important helper logic works.

Current test file:

```text
tests/test_plate_utils.py
```

It checks:

- standard Indian plate formats
- BH-series formatting
- noisy OCR cleanup
- invalid state-code rejection

Run:

```bash
python3 -m pytest tests/test_plate_utils.py -q
```
