import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(page_title="Scanner LJK Per 10 Nomor - SMP YPI Pulogadung", layout="wide")

# ---------------------------------------------------------
# CSS: BINGKAI PANDUAN KAMERA PER KOLOM (10 NOMOR)
# ---------------------------------------------------------
st.markdown("""
    <style>
    div[data-testid="stCameraInput"] {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stCameraInput"]::before {
        content: "🎯 PASIKAN 1 KOLOM (10 NOMOR) DI DALAM BINGKAI";
        position: absolute;
        top: 4%;
        left: 10%;
        width: 80%;
        height: 90%;
        border: 3px dashed #00FF00;
        border-radius: 12px;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.40);
        z-index: 99;
        pointer-events: none;
        color: #00FF00;
        font-weight: bold;
        font-size: 13px;
        text-align: center;
        padding-top: 10px;
        background: rgba(0, 255, 0, 0.05);
        box-sizing: border-box;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📱 Pemindai LJK Modul (10 Soal per Foto)")
st.caption("Khusus Format LJK SMP YPI Pulogadung (4 Blok: Soal 1-10, 11-20, 21-30, 31-40)")

# ---------------------------------------------------------
# PREPROCESSING & MEMPROSES 1 KOLOM (10 NOMOR)
# ---------------------------------------------------------
def prepare_column_image(file_bytes):
    raw_pil = Image.open(file_bytes)
    try:
        raw_pil = ImageOps.exif_transpose(raw_pil)
    except Exception:
        pass
    
    # Resize ringan khusus 1 kolom (300 x 600 px)
    w, h = raw_pil.size
    if max(w, h) > 800:
        scale = 800.0 / float(max(w, h))
        raw_pil = raw_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
    return np.array(raw_pil.convert('RGB'))

def process_single_column(image_np, q_range, key_answers, sensitivity_delta=15):
    """Memproses 1 foto yang berisi 1 kolom (10 nomor)"""
    # Standardisasi dimensi gambar ke 250x500 px
    resized = cv2.resize(image_np, (250, 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    h, w = binary.shape
    row_h = h / 10.0      # 10 Baris Soal
    sub_col_w = w / 5.0  # Sub-kolom 0: No, Sub-kolom 1-4: A, B, C, D
    
    options = ['A', 'B', 'C', 'D']
    detected_answers = {}
    annotated_img = resized.copy()
    
    for r_idx, q_num in enumerate(q_range):
        row_y_start = r_idx * row_h
        densities = []
        cell_coords = []
        
        for opt_idx in range(4):
            x1 = int(((opt_idx + 1) * sub_col_w) + (sub_col_w * 0.15))
            x2 = int(((opt_idx + 2) * sub_col_w) - (sub_col_w * 0.15))
            y1 = int(row_y_start + (row_h * 0.15))
            y2 = int(row_y_start + row_h - (row_h * 0.15))
            
            cell_coords.append((x1, y1, x2, y2))
            
            cell = binary[y1:y2, x1:x2]
            pixel_count = cv2.countNonZero(cell) if cell.size > 0 else 0
            densities.append(pixel_count)
            
        max_val = max(densities)
        max_idx = densities.index(max_val)
        
        other_vals = [v for i, v in enumerate(densities) if i != max_idx]
        avg_others = np.mean(other_vals) if len(other_vals) > 0 else 0
        
        selected_option = "-"
        if (max_val - avg_others) > sensitivity_delta:
            selected_option = options[max_idx]
            
        detected_answers[q_num] = selected_option
        
        for opt_idx, (x1, y1, x2, y2) in enumerate(cell_coords):
            if opt_idx == max_idx and selected_option != "-":
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
                
    return detected_answers, annotated_img

# ---------------------------------------------------------
# SIDEBAR KUNCI JAWABAN & SENSITIVITAS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Kunci Jawaban (40 Soal)")
key_dict = {}
cols = st.sidebar.columns(2)
for i in range(1, 41):
    col_target = cols[0] if i <= 20 else cols[1]
    key_dict[i] = col_target.selectbox(f"Soal {i}", ['A', 'B', 'C', 'D'], index=0, key=f"k_{i}")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Sensitivitas Coretan")
delta_thresh = st.sidebar.slider("Kontras Kehitaman Coretan (Delta)", 5, 50, 15, 1)

# Initialize Session State untuk menyimpan jawaban dari 4 blok
if "all_answers" not in st.session_state:
    st.session_state.all_answers = {}

# ---------------------------------------------------------
# TAB 4 BLOK PENAMBILAN FOTO (PER 10 NOMOR)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Soal 1 – 10", 
    "📸 Soal 11 – 20", 
    "📸 Soal 21 – 30", 
    "📸 Soal 31 – 40"
])

blocks = [
    (tab1, range(1, 11), "b1_cam"),
    (tab2, range(11, 21), "b2_cam"),
    (tab3, range(21, 31), "b3_cam"),
    (tab4, range(31, 41), "b4_cam")
]

for tab, q_range, cam_key in blocks:
    with tab:
        st.write(f"**Ambil Foto Kolom Soal Nomor {q_range[0]} s.d. {q_range[-1]}**")
        cam_photo = st.camera_input(f"Kamera Soal {q_range[0]}-{q_range[-1]}", key=cam_key)
        
        if cam_photo is not None:
            img_np = prepare_column_image(cam_photo)
            answers, annotated_img = process_single_column(img_np, q_range, key_dict, delta_thresh)
            
            # Simpan hasil pembacaan ke session_state
            for q_num, ans in answers.items():
                st.session_state.all_answers[q_num] = ans
                
            c_img, c_info = st.columns([1, 2])
            with c_img:
                st.image(annotated_img, caption=f"Hasil Scan Soal {q_range[0]}-{q_range[-1]}", use_container_width=True)
            with c_info:
                st.success(f"✅ Blok Soal {q_range[0]}–{q_range[-1]} Berhasil Terbaca!")
                st.json({f"Soal {k}": v for k, v in answers.items()})

# ---------------------------------------------------------
# RINGKASAN REKAPITULASI NILAI AKHIR (1 - 40)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Hasil Rekapitulasi Nilai Akhir (Soal 1 s.d. 40)")

score_correct = 0
results = []

for q_num in range(1, 41):
    user_ans = st.session_state.all_answers.get(q_num, "-")
    key_ans = key_dict.get(q_num, "A")
    
    is_correct = (user_ans == key_ans)
    if is_correct:
        score_correct += 1
        
    results.append({
        "No": q_num,
        "Jawaban Siswa": user_ans,
        "Kunci Jawaban": key_ans,
        "Status": "✅ Benar" if is_correct else ("❌ Salah" if user_ans != "-" else "⚪ Belum Di-scan")
    })

final_score = (score_correct / 40.0) * 100.0

col_metric1, col_metric2, col_metric3 = st.columns(3)
col_metric1.metric("Nilai Akhir", f"{final_score:.1f}")
col_metric2.metric("Jumlah Benar", f"{score_correct} / 40")
col_metric3.metric("Soal Ter-scan", f"{len(st.session_state.all_answers)} / 40")

df_res = pd.DataFrame(results)
st.dataframe(df_res, height=350, use_container_width=True)
