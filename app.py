import streamlit as st
import cv2
import numpy as np
import pandas as pd
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Live Scanner LJK - SMP YPI Pulogadung", layout="wide")

st.title("⚡ Pemindai LJK Real-Time (Tanpa Tombol Foto)")
st.caption("Arahkan kamera HP ke LJK — sistem akan mendeteksi dan mengoreksi secara otomatis secara live.")

# ---------------------------------------------------------
# SIDEBAR: OPSI JUMLAH SOAL & KUNCI JAWABAN
# ---------------------------------------------------------
st.sidebar.header("📋 Mode Pengerjaan")
num_questions = st.sidebar.radio(
    "Pilih Jumlah Soal Pilihan Ganda:",
    options=[40, 30],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header(f"⚙️ Kunci Jawaban ({num_questions} Soal)")
key_dict = {}
cols = st.sidebar.columns(2)
for i in range(1, num_questions + 1):
    col_target = cols[0] if i <= (num_questions // 2) else cols[1]
    key_dict[i] = col_target.selectbox(f"Soal {i}", ['A', 'B', 'C', 'D'], index=0, key=f"k_{i}")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Sensitivitas Silang")
delta_thresh = st.sidebar.slider("Sensitivitas Kehitaman Coretan", 5, 50, 15, 1)

# Save configurations to session state for WebRTC thread access
st.session_state['num_questions'] = num_questions
st.session_state['key_dict'] = key_dict
st.session_state['delta_thresh'] = delta_thresh

# ---------------------------------------------------------
# LOGIKA PEMBACAAN FRAME LIVE (REAL-TIME OPENCV)
# ---------------------------------------------------------
class LJKVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_score = None
        self.latest_correct = None
        self.latest_answers = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr2rgb")
        h, w, _ = img.shape

        # Downsample ringan untuk performa real-time tinggi
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            img_small = img.copy()

        sh, sw, _ = img_small.shape
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bh / float(bw) if bw > 0 else 0
            area = bw * bh
            
            if (sw * sh * 0.02) < area < (sw * sh * 0.25) and 1.2 <= aspect_ratio <= 3.8:
                boxes.append((x, y, bw, bh))
                
        boxes = sorted(boxes, key=lambda b: b[0])

        # Menggambar Bingkai Panduan Live pada Video
        annotated_img = img_small.copy()
        
        if len(boxes) >= 4:
            selected_boxes = boxes[:4]
            # Gambar kotak hijau pada 4 kolom yang terdeteksi
            for x, y, bw, bh in selected_boxes:
                cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                
            cv2.putText(annotated_img, "LJK TERDETEKSI - DITINGKATKAN", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Tampilkan overlay bingkai bantuan jika belum terisolasi penuh
            cv2.putText(annotated_img, "ARAHKAN KAMERA KEPADA 4 KOLOM LJK", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(annotated_img, format="rgb24")

# ---------------------------------------------------------
# STREAMER WEBRTC KAMERA LIVE
# ---------------------------------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="ljk-live-scanner",
    video_processor_factory=LJKVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
)

st.info("💡 **Petunjuk:** Aktifkan izin kamera. Cukup dekatkan LJK ke kamera tanpa perlu menekan tombol ambil foto.")
