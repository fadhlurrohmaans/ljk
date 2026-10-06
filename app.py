import streamlit as st
import cv2
import numpy as np
import pandas as pd
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(
    page_title="Live Scanner LJK - SMP YPI Pulogadung", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# CSS: RESPONSIF FULL SCREEN HP & PANDUAN BINGKAI OVERLAY
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Minimalkan padding Streamlit agar video kamera memenuhi layar HP */
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
        max-width: 100% !important;
    }
    
    /* Buat pemutar video WebRTC tampil full width & responsif */
    div[data-testid="stWebRtc"] {
        width: 100% !important;
        position: relative !important;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stWebRtc"] video {
        width: 100% !important;
        height: auto !important;
        max-height: 80vh !important;
        object-fit: cover !important;
    }

    /* Sembunyikan elemen bawaan yang tidak diperlukan saat scan */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Live Scanner LJK SMP YPI")

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

# Save configurations to session state
st.session_state['num_questions'] = num_questions
st.session_state['key_dict'] = key_dict
st.session_state['delta_thresh'] = delta_thresh

# ---------------------------------------------------------
# LOGIKA PEMBACAAN FRAME LIVE DENGAN ANCHOR GUIDE
# ---------------------------------------------------------
class LJKVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr2rgb")
        h, w, _ = img.shape

        # Downsample untuk performa tinggi
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
        annotated_img = img_small.copy()

        # ---------------------------------------------------------
        # ANCHOR GUIDES (PANDUAN VISUAL SIKU & KOLOM TARGET)
        # ---------------------------------------------------------
        is_detected = (len(boxes) >= 4)
        guide_color = (0, 255, 0) if is_detected else (0, 215, 255) # Hijau jika pas, Kuning jika belum
        
        # 1. Gambar Siku-Siku Kamera (Corner Anchors)
        margin_x, margin_y = int(sw * 0.05), int(sh * 0.15)
        box_w, box_h = sw - (2 * margin_x), sh - (2 * margin_y)
        corner_len = 30
        
        # Top-Left Corner
        cv2.line(annotated_img, (margin_x, margin_y), (margin_x + corner_len, margin_y), guide_color, 3)
        cv2.line(annotated_img, (margin_x, margin_y), (margin_x, margin_y + corner_len), guide_color, 3)
        # Top-Right Corner
        cv2.line(annotated_img, (margin_x + box_w, margin_y), (margin_x + box_w - corner_len, margin_y), guide_color, 3)
        cv2.line(annotated_img, (margin_x + box_w, margin_y), (margin_x + box_w, margin_y + corner_len), guide_color, 3)
        # Bottom-Left Corner
        cv2.line(annotated_img, (margin_x, margin_y + box_h), (margin_x + corner_len, margin_y + box_h), guide_color, 3)
        cv2.line(annotated_img, (margin_x, margin_y + box_h), (margin_x, margin_y + box_h - corner_len), guide_color, 3)
        # Bottom-Right Corner
        cv2.line(annotated_img, (margin_x + box_w, margin_y + box_h), (margin_x + box_w - corner_len, margin_y + box_h), guide_color, 3)
        cv2.line(annotated_img, (margin_x + box_w, margin_y + box_h), (margin_x + box_w, margin_y + box_h - corner_len), guide_color, 3)

        # 2. Gambar 4 Kolom Panduan Maya (Column Anchor Slots)
        col_w = box_w / 4.0
        for i in range(4):
            cx1 = int(margin_x + (i * col_w) + (col_w * 0.1))
            cx2 = int(margin_x + ((i + 1) * col_w) - (col_w * 0.1))
            cy1 = int(margin_y + (box_h * 0.15))
            cy2 = int(margin_y + (box_h * 0.85))
            
            # Gambar garis vertikal bantuan tipis
            cv2.rectangle(annotated_img, (cx1, cy1), (cx2, cy2), (180, 180, 180), 1)

        # 3. Status Hasil Deteksi
        if is_detected:
            selected_boxes = boxes[:4]
            for x, y, bw, bh in selected_boxes:
                cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                
            cv2.putText(annotated_img, "PAS! LJK TERDETEKSI", (margin_x, margin_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_img, "PAS-KAN 4 KOLOM DI DALAM BINGKAI", (margin_x, margin_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

        return av.VideoFrame.from_ndarray(annotated_img, format="rgb24")

# ---------------------------------------------------------
# STREAMER WEBRTC KAMERA LIVE
# ---------------------------------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="ljk-live-scanner",
    video_processor_factory=LJKVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
)

st.info("💡 **Petunjuk:** Pas-kan 4 kolom tabel LJK ke dalam 4 kotak panduan maya di layar.")
