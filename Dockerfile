# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the image small while having all needed build tools.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# tesseract-ocr: OCR binary required by pytesseract
# libgl1: required by OpenCV
# libglib2.0-0: required by EasyOCR / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer.
# If only your code changes, Docker re-uses the installed packages layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        pytesseract \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8501 → Streamlit app
# 8000 → FastAPI REST API
EXPOSE 8501 8000

# ── Default: start the Streamlit app ─────────────────────────────────────────
# Override CMD in docker-compose to start the API instead.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
