"""
Indian ANPR System — Streamlit App
YOLOv11 + EasyOCR + Tesseract
Supports: Standard · HSRP · 2-Line · BH Series
"""
from __future__ import annotations

import os
import time
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from ocr_engine import read_plate
from plate_utils import is_valid_indian_plate, format_plate

st.set_page_config(
    page_title="ANPR India – Smart Plate Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(1200px 600px at 8% -10%, rgba(56,139,253,0.16) 0%, rgba(13,17,23,0.0) 55%),
          radial-gradient(900px 500px at 110% 0%, rgba(63,185,80,0.12) 0%, rgba(13,17,23,0.0) 50%),
          #0d1117;
    }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0.6rem; }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1.2rem;
        max-width: 1360px;
    }
    [data-testid="stImage"] {
        border: 1px solid #30363d;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
    }
    [data-testid="stImage"] img {
        max-height: 400px;
        object-fit: contain;
    }
    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .sec-hdr {
        font-size: 0.85rem; font-weight: 700; color: #58a6ff; text-transform: uppercase;
        letter-spacing: 0.1em; border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-bottom: 14px;
    }
    .plate-badge {
        display: inline-block; font-family: 'Courier New', monospace; font-size: 1.25rem;
        font-weight: 800; letter-spacing: 0.15em; color: #ffffff;
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        padding: 6px 18px; border-radius: 8px; margin: 4px 0;
        box-shadow: 0 3px 12px rgba(56, 139, 253, 0.4);
    }
    .plate-badge-warn {
        display: inline-block; font-family: 'Courier New', monospace; font-size: 1.1rem;
        font-weight: 700; letter-spacing: 0.12em; color: #ffffff;
        background: linear-gradient(135deg, #9e6a03 0%, #d29922 100%);
        padding: 5px 14px; border-radius: 8px; margin: 4px 0;
        box-shadow: 0 3px 10px rgba(210, 153, 34, 0.35);
    }
    .conf-pill {
        display: inline-block; font-size: 0.78rem; font-weight: 600; color: #8b949e;
        background: #21262d; border: 1px solid #30363d; border-radius: 20px;
        padding: 2px 10px; margin-left: 8px; vertical-align: middle;
    }
    .tag-2line {
        display: inline-block; font-size: 0.7rem; font-weight: 600; color: #3fb950;
        background: rgba(63, 185, 80, 0.12); border: 1px solid rgba(63, 185, 80, 0.3);
        border-radius: 12px; padding: 1px 9px; margin-left: 6px; vertical-align: middle;
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #30363d !important; border-radius: 10px !important; background: #0d1117 !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: #388bfd !important; }
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid #30363d !important;
    }
    .stButton > button:hover {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.20) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_BASE = Path(__file__).parent
MODEL_PATH = _BASE / "best.pt"

@st.cache_resource(show_spinner="Loading YOLOv11 + EasyOCR…")
def load_models():
    import easyocr
    yolo_m = YOLO(str(MODEL_PATH))
    ocr_r = easyocr.Reader(["en"], gpu=False, verbose=False)
    return yolo_m, ocr_r

yolo_model, ocr_reader = load_models()

if "log" not in st.session_state:
    st.session_state.log = []
if "proc_ms" not in st.session_state:
    st.session_state.proc_ms = []

def detect_on_frame(bgr: np.ndarray, yolo_conf: float, iou_thr: float,
                    strict: bool, use_tess: bool) -> tuple[np.ndarray, list[dict]]:
    yolo_out = yolo_model(bgr, conf=yolo_conf, iou=iou_thr, max_det=15,
                          imgsz=640, verbose=False)[0]
    found = []
    for box in yolo_out.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        # Keep more right-edge context because the final digit is often near the border.
        pad_left = max(2, int(bw * 0.05))
        pad_right = max(3, int(bw * 0.14))
        pad_top = max(2, int(bh * 0.12))
        pad_bottom = max(2, int(bh * 0.14))
        x1, y1 = max(0, x1 - pad_left), max(0, y1 - pad_top)
        x2, y2 = min(bgr.shape[1], x2 + pad_right), min(bgr.shape[0], y2 + pad_bottom)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        result = read_plate(crop, ocr_reader, use_tesseract=use_tess)
        text = result["text"]
        conf = result["confidence"]
        two_line = result["two_line"]
        raw = result["raw"]
        valid = is_valid_indian_plate(text)
        if strict and not valid:
            continue
        color = (0, 215, 55) if valid else (0, 165, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 3)
        label = f"{text}  {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.rectangle(bgr, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, cv2.FILLED)
        cv2.putText(bgr, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)
        if not text:
            continue
        found.append({"text": text, "conf": conf, "confidence": conf,
                      "two_line": two_line, "valid": valid, "raw": raw})
    return bgr, found

def _log_detection(det: dict, source: str, time_label: str) -> None:
    existing = {d["plate"] for d in st.session_state.log}
    conf = det.get("confidence", det.get("conf", 0.0))
    if det["text"] and det["text"] not in existing:
        st.session_state.log.append({
            "time": time_label,
            "plate": det["text"],
            "formatted": format_plate(det["text"]),
            "confidence": conf,
            "two_line": det["two_line"],
            "valid": det["valid"],
            "source": source,
        })

def _plate_html(
    text: str,
    conf: float = 0.0,
    two_line: bool = False,
    valid: bool = False,
    confidence: float | None = None,
    **_: object,
) -> str:
    if confidence is not None:
        conf = confidence
    badge_cls = "plate-badge" if valid else "plate-badge-warn"
    tl_tag = '<span class="tag-2line">2-LINE</span>' if two_line else ""
    return (f'<span class="{badge_cls}">{text}</span>'
            f'<span class="conf-pill">{conf * 100:.1f}% conf</span>{tl_tag}')

with st.sidebar:
    st.markdown("<h2 style='color:#58a6ff;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    st.divider()
    yolo_conf = st.slider("Detection Confidence", 0.10, 0.90, 0.25, 0.05,
                          help="Higher values prevent false positives (like bumpers) but may miss blurry plates.")
    iou_thresh = st.slider("IoU Threshold", 0.10, 0.90, 0.45, 0.05,
                           help="Prevents duplicate boxes on the same plate. 0.45 is the research standard.")
    st.markdown("**OCR Options**")
    use_tess = st.toggle("Dual Engine (EasyOCR + Tesseract)", value=True,
                         help="Runs two engines for higher accuracy, but takes 2x longer.")
    strict_val = st.toggle("Strict Indian Format", value=True,
                           help="Hides results that do not mathematically match Indian plate rules.")
    show_raw = st.toggle("Show Raw OCR", value=False,
                         help="Shows the uncleaned text from the OCR engine before Python fixes it.")
    st.divider()
    if st.button("🗑️ Clear Log", use_container_width=True, type="secondary"):
        st.session_state.log.clear()
        st.session_state.proc_ms.clear()
        st.rerun()

st.markdown("""
<div style="display:flex; align-items:center; gap:12px;">
  <span style="font-size:2.2rem;">🇮🇳</span>
  <div>
    <h1 style="margin:0; font-size:1.9rem;">Indian ANPR System</h1>
    <p style="margin:0; color:#6e7681; font-size:0.88rem;">
      YOLOv11 · EasyOCR · Tesseract · Standard · HSRP · 2-Line · BH Series
    </p>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

tab_img, tab_vid = st.tabs(["📷 Image", "🎬 Video"])

with tab_img:
    col_left, col_right = st.columns([1.1, 1], gap="large")
    
    with col_left:
        st.markdown('<div class="sec-hdr">Input Image</div>', unsafe_allow_html=True)
        img_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png", "bmp"], key="img_upload")
        
        # We define the placeholder here so the image always renders on the left
        image_placeholder = st.empty()
        
        if img_file:
            image_placeholder.image(Image.open(img_file), use_container_width=True)
            img_file.seek(0)
            
    with col_right:
        st.markdown('<div class="sec-hdr">Action & Results</div>', unsafe_allow_html=True)
        
        if img_file:
            detect_clicked = st.button("🔍 Detect License Plate", type="primary", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            detect_clicked = False
            
        res_placeholder = st.container()
        
        st.markdown('<div class="sec-hdr" style="margin-top:30px;">Session Stats</div>', unsafe_allow_html=True)
        stats_placeholder = st.empty()
        
    def render_stats():
        _log = st.session_state.log
        if not _log:
            stats_placeholder.info("No plates logged yet. Run a detection to see stats.")
            return
        _n = len(_log)
        _nv = sum(1 for d in _log if d.get("valid"))
        _ac = float(np.mean([d["confidence"] for d in _log]) * 100)
        _ams = float(np.mean(st.session_state.proc_ms)) if st.session_state.proc_ms else 0.0
        
        with stats_placeholder.container():
            c1, c2 = st.columns(2)
            c1.metric("Total Plates", _n)
            c2.metric("Valid Format", _nv)
            c1.metric("Avg Confidence", f"{_ac:.1f}%")
            c2.metric("Avg Processing", f"{_ams:.0f}ms")

    # Initial render of the stats box
    render_stats()
            
    if img_file and detect_clicked:
        raw_bytes = img_file.read()
        bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        with col_left:
            with st.spinner("Detecting and Reading Plate..."):
                t0 = time.perf_counter()
                annotated, found = detect_on_frame(bgr, yolo_conf, iou_thresh, strict_val, use_tess)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                
        st.session_state.proc_ms.append(elapsed_ms)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # 🚀 SWAP the image: Replace the original with the scanned/annotated one
        image_placeholder.image(rgb, use_container_width=True)
        
        # Fill results in the right column
        with res_placeholder:
            if found:
                st.success(f"Found **{len(found)}** plate(s) in **{elapsed_ms:.0f}ms**")
                for det in found:
                    st.markdown(_plate_html(**det), unsafe_allow_html=True)
                    if show_raw:
                        st.caption(f"Raw: `{det['raw']}`")
                    _log_detection(det, "image", datetime.now().strftime("%H:%M:%S"))
            else:
                st.warning("No plates found in this image. Try lowering the Detection Confidence in settings.")
                
        # Instantly update stats after detection
        render_stats()

with tab_vid:
    st.markdown('<div class="sec-hdr">Input Video</div>', unsafe_allow_html=True)
    vid_col, opt_col = st.columns([3, 1], gap="large")
    with opt_col:
        st.markdown("**Options**")
        frame_skip = st.slider("Skip frames", 1, 15, 4, 1)
        max_unique = st.number_input("Max plates", 1, 500, 100)
    with vid_col:
        vid_file = st.file_uploader("Drag & drop", type=["mp4", "avi", "mov", "mkv"],
                                     key="vid_upload")
    if vid_file and st.button("▶️ Process", type="primary", key="btn_proc_vid"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vid_file.read())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_idx = 0
        seen = set()
        preview = st.empty()
        prog_bar = st.progress(0.0, text="Starting…")
        new_plate = st.empty()
        t0 = time.perf_counter()
        while cap.isOpened() and len(seen) < max_unique:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue
            annotated, found = detect_on_frame(frame, yolo_conf, iou_thresh,
                                            strict_val, use_tess)
            for det in found:
                if det["text"] and det["text"] not in seen:
                    seen.add(det["text"])
                    new_plate.markdown(f"🆕 {_plate_html(**det)}",
                                      unsafe_allow_html=True)
                    _log_detection(det, "video", f"Frame {frame_idx}")
            preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB",
                         use_container_width=True)
            prog = min(frame_idx / total_frames, 1.0)
            prog_bar.progress(prog, text=f"Frame {frame_idx}/{total_frames}")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        st.session_state.proc_ms.append(elapsed_ms)
        cap.release()
        os.unlink(tmp.name)
        prog_bar.progress(1.0, text="Complete ✓")
        st.success(f"**{len(seen)}** unique plates in **{elapsed_ms/1000:.1f}s**")

st.divider()
st.markdown("## 📋 Detection Log")
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    display_df = df[["time", "formatted", "confidence", "two_line", "valid", "source"]].copy()
    display_df.columns = ["Time", "Plate", "Conf %", "2-Line", "Valid", "Source"]
    display_df["Conf %"] = (display_df["Conf %"] * 100).round(1)
    tbl_col, dl_col = st.columns([5, 1], gap="medium")
    with tbl_col:
        st.dataframe(display_df, use_container_width=True,
                    height=min(480, 60 + len(display_df) * 36))
    with dl_col:
        st.markdown("### Export")
        export_df = df[["time", "plate", "formatted", "confidence", "two_line", "valid",
                        "source"]].copy()
        export_df["confidence"] = (export_df["confidence"] * 100).round(2)
        export_df.columns = ["Time", "Raw", "Formatted", "Conf %", "2-Line", "Valid", "Source"]
        st.download_button(label="⬇️ CSV",
                          data=export_df.to_csv(index=False).encode(),
                          file_name=f"ANPR_{datetime.now():%Y%m%d_%H%M%S}.csv",
                          mime="text/csv", use_container_width=True)
        if st.button("🗑️ Clear", use_container_width=True, type="secondary"):
            st.session_state.log.clear()
            st.session_state.proc_ms.clear()
            st.rerun()
else:
    st.info("No plates logged yet.")
