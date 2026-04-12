# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### macOS

```bash
# 1. Prerequisites
brew install tesseract

# 2. Navigate to project
cd License-Plate-Recognition-app

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies (offline-friendly)
pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Run app
streamlit run app.py
```

**Output**: Open http://localhost:8501 in your browser.

---

### Linux (Ubuntu/Debian)

```bash
# 1. Install Tesseract
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# 2. Setup & run
cd License-Plate-Recognition-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

### Windows (PowerShell)

```powershell
# 1. Install Tesseract (via installer)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# 2. Setup
cd License-Plate-Recognition-app
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

---

## 📷 First Test

1. **Open browser** → http://localhost:8501
2. **Upload image** → Select a car photo with visible plate
3. **Adjust settings** if needed:
   - Start with default (Confidence: 0.20, IoU: 0.45)
4. **Click Detect** → View results
5. **Check CSV export** → Download detection log

---

## ⚡ Performance Tips

### For CPU (slower)
- Keep detection confidence **≥0.20**
- Frame-skip video **≥4** (process 1 in 5 frames)
- Resize images to **720p** before upload

### For GPU (faster)
If you have CUDA 11.8+:
```bash
pip install torch torchvision torch-cuda

# Enable in code
yolo_model = YOLO("best.pt").cuda()
ocr_reader = easyocr.Reader(["en"], gpu=True)
```

---

## 🔧 Troubleshooting

### `Port already in use`
```bash
# Kill existing Streamlit process
lsof -ti:8501 | xargs kill -9
streamlit run app.py --server.port 8502  # Use different port
```

### `Tesseract not found`
```bash
# Verify installation
tesseract --version

# If not in PATH on Windows, set manually in pytesseract:
import pytesseract
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### `CUDA out of memory`
```python
# Reduce batch size
yolo_model(bgr, batch=1)
```

---

## 📚 Next Steps

1. Read [README.md](README.md) for full documentation
2. Explore [ARCHITECTURE.md](ARCHITECTURE.md) to understand the pipeline
3. Check [CONTRIBUTING.md](CONTRIBUTING.md) for code guidelines
4. Try [API usage examples](README.md#-api--python-usage)

---

## 🎯 Common Use Cases

### Batch Process Local Folder
```python
import cv2
import glob
from ultralytics import YOLO
from ocr_engine import read_plate
import easyocr

yolo = YOLO("best.pt")
reader = easyocr.Reader(["en"], gpu=False)

for img_path in glob.glob("sample_images/*.jpg"):
    bgr = cv2.imread(img_path)
    results = yolo(bgr, conf=0.20)[0]
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = bgr[y1:y2, x1:x2]
        ocr = read_plate(crop, reader)
        print(f"{img_path}: {ocr['text']}")
```

### Process Video File
```python
import cv2
from ultralytics import YOLO
from ocr_engine import read_plate
import easyocr

cap = cv2.VideoCapture("video.mp4")
yolo = YOLO("best.pt")
reader = easyocr.Reader(["en"], gpu=False)

frame_idx = 0
unique_plates = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    if frame_idx % 4 != 0:  # Process every 4th frame
        continue
    
    results = yolo(frame, conf=0.20)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        ocr = read_plate(crop, reader)
        if ocr['text'] not in unique_plates:
            unique_plates.add(ocr['text'])
            print(f"New: {ocr['text']} @ frame {frame_idx}")

cap.release()
print(f"Total unique plates: {len(unique_plates)}")
```

---

**Happy plate recognition! 🇮🇳**
