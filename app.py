import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Auto-Scan LJK - SMP YPI Pulogadung", layout="wide")

st.markdown("""
    <style>
    div[data-testid="stCameraInput"] {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stCameraInput"]::before {
        content: "🎯 PASTIKAN SOAL 1 (KIRI ATAS) DAN SOAL 40 (KANAN BAWAH) MASUK DALAM BINGKAI";
        position: absolute;
        top: 6%;
        left: 4%;
        width: 92%;
        height: 84%;
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

st.title("📱 Pemindai LJK Otomatis (Anchor Soal 1 & 40)")
st.caption("Khusus Format LJK SMP YPI Pulogadung (40 Soal Pilihan Ganda)")

# ---------------------------------------------------------
# DETEKSI ANCHOR OTOMATIS (SOAL 1 & SOAL 40)
# ---------------------------------------------------------
def detect_q1_q40_anchors(image_np):
    """
    Mendeteksi struktur garis tabel LJK untuk mengunci:
    - Patokan 1: Pojok Kiri Atas (Soal 1) -> (xmin, ymin)
    - Patokan 2: Pojok Kanan Bawah (Soal 40) -> (xmax, ymax)
    """
    h, w, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Thresholding untuk ekstraksi garis hitam LJK
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Deteksi Garis Horizontal & Vertikal menggunakan Morphological Ops
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    
    # Gabungkan struktur garis tabel
    table_grid = cv2.add(horiz, vert)
    
    # Cari kontur garis di area tengah gambar (Pilihan Ganda berada di 20%-65% tinggi foto)
    roi_mask = np.zeros_like(table_grid)
    roi_mask[int(h * 0.20):int(h * 0.65), int(w * 0.02):int(w * 0.98)] = 255
    masked_grid = cv2.bitwise_and(table_grid, table_grid, mask=roi_mask)
    
    contours, _ = cv2.findContours(masked_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > 30 and bh > 10:  # Filter noise kecil
            valid_boxes.append((x, y, x + bw, y + bh))
            
    if len(valid_boxes) > 0:
        # Tentukan Bounding Box terluar yang membungkus seluruh tabel (Soal 1 s.d. Soal 40)
        xmin = min([b[0] for b in valid_boxes])
        ymin = min([b[1] for b in valid_boxes])
        xmax = max([b[2] for b in valid_boxes])
        ymax = max([b[3] for b in valid_boxes])
        
        # Crop & Resize ke resolusi standar (1200 x 500 px)
        cropped = image_np[ymin:ymax, xmin:xmax]
        resized = cv2.resize(cropped, (1200, 500))
        return resized, True, (xmin, ymin, xmax, ymax)
    else:
        # Fallback Crop jika foto sangat redup
        crop = image_np[int(h * 0.25):int(h * 0.55), int(w * 0.04):int(w * 0.96)]
        resized = cv2.resize(crop, (1200, 500))
        return resized, False, (0, 0, 0, 0)

# ---------------------------------------------------------
# EVALUASI MATRIX JAWABAN (10 BARIS x 4 KOLOM UTAMA)
# ---------------------------------------------------------
def process_anchored_grid(warped_img, key_answers, sensitivity=110, min_pixels=45):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY_INV)
    
    h, w = binary.shape
    col_w = w / 4.0   # 4 Kolom Utama (1-10, 11-20, 21-30, 31-40)
    row_h = h / 10.0  # 10 Baris Soal
    
    col_ranges = [
        range(1, 11),   # Col 1: Soal 1-10
        range(11, 21),  # Col 2: Soal 11-20
        range(21, 31),  # Col 3: Soal 21-30
        range(31, 41)   # Col 4: Soal 31-40
    ]
    options = ['A', 'B', 'C', 'D']
    
    detected_answers = {}
    annotated_img = warped_img.copy()
    
    for c_idx, q_range in enumerate(col_ranges):
        col_x_start = c_idx * col_w
        
        for r_idx, q_num in enumerate(q_range):
            row_y_start = r_idx * row_h
            sub_col_w = col_w / 5.0  # 1 Sub-kolom No + 4 Sub-kolom Opsi (A, B, C, D)
            
            max_pixels = 0
            selected_option = "-"
            
            for opt_idx, opt_label in enumerate(options):
                # Opsi A-D berada di sub-kolom ke 2-5 (indeks 1-4)
                x1 = int(col_x_start + ((opt_idx + 1) * sub_col_w) + (sub_col_w * 0.18))
                x2 = int(col_x_start + ((opt_idx + 2) * sub_col_w) - (sub_col_w * 0.18))
                y1 = int(row_y_start + (row_h * 0.18))
                y2 = int(row_y_start + row_h - (row_h * 0.18))
                
                cell = binary[y1:y2, x1:x2]
                pixel_count = cv2.countNonZero(cell) if cell.size > 0 else 0
                
                color = (0, 255, 0) if pixel_count > min_pixels else (200, 200, 200)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 1)
                
                if pixel_count > min_pixels and pixel_count > max_pixels:
                    max_pixels = pixel_count
                    selected_option = opt_label
                    
            detected_answers[q_num] = selected_option

    # Hitung Nilai
    score_correct = 0
    results = []
    
    for q_num in range(1, 41):
        user_ans = detected_answers.get(q_num, "-")
        key_ans = key_answers.get(q_num, "A")
        
        is_correct = (user_ans == key_ans)
        if is_correct:
            score_correct += 1
            
        results.append({
            "No": q_num,
            "Jawaban": user_ans,
            "Kunci": key_ans,
            "Hasil": "✅" if is_correct else "❌"
        })
        
    final_score = (score_correct / 40.0) * 100.0
    return final_score, score_correct, results, annotated_img

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.sidebar.header("⚙️ Kunci Jawaban (40 Soal)")
key_dict = {}
cols = st.sidebar.columns(2)
for i in range(1, 41):
    col_target = cols[0] if i <= 20 else cols[1]
    key_dict[i] = col_target.selectbox(f"Soal {i}", ['A', 'B', 'C', 'D'], index=0, key=f"k_{i}")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Ambang Toleransi")
threshold_val = st.sidebar.slider("Kehitaman Pensil/Coretan", 50, 200, 110)
min_pixel_val = st.sidebar.slider("Ukuran Coretan Minimal", 20, 150, 45)

option_input = st.radio("Pilih Metode Pemindaian:", ["📷 Pakai Kamera HP Live", "📁 Unggah File Foto"], horizontal=True)

uploaded_file = None
if option_input == "📷 Pakai Kamera HP Live":
    uploaded_file = st.camera_input("Arahkan kamera ke kertas LJK")
else:
    uploaded_file = st.file_uploader("Unggah file foto LJK (JPG / PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file)
        img_np = np.array(pil_image.convert('RGB'))
        
        # 1. Deteksi Anchor Soal 1 & Soal 40
        anchored_grid, is_auto, coords = detect_q1_q40_anchors(img_np)
        
        # 2. Proses Evaluasi Jawaban
        score, correct_count, results, annotated_grid = process_anchored_grid(
            anchored_grid, key_dict, threshold_val, min_pixel_val
        )
        
        if is_auto:
            st.success(f"✅ Titik Acuan Terkunci! (Patokan Soal 1: Kiri-Atas [{coords[0]},{coords[1]}], Soal 40: Kanan-Bawah [{coords[2]},{coords[3]}])")
        else:
            st.info("ℹ️ Menggunakan pemotongan area estimasi standar.")

        c1, c2 = st.columns([1.2, 1])
        
        with c1:
            st.subheader("🔍 Matriks Terdeteksi (Soal 1 s.d. 40)")
            st.image(annotated_grid, use_container_width=True, caption="Kotak Hijau = Coretan 'X' Terbaca Sistem")
            
        with c2:
            st.subheader("📊 Ringkasan Nilai")
            st.metric("Nilai Akhir", f"{score:.1f}")
            st.write(f"**Jumlah Benar:** {correct_count} / 40 Soal")
            
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, height=350, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memproses gambar: {str(e)}")
