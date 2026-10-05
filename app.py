import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Koreksi LJK Otomatis - SMP YPI Pulogadung", layout="wide")

st.title("📋 Pemindai & Koreksi LJK Otomatis")
st.caption("Khusus Format LJK SMP YPI Pulogadung (40 Soal Pilihan Ganda - Tanda Silang)")

# ---------------------------------------------------------
# FUNGSI PENGOLAHAN CITRA & DETEKSI TANDA SILANG
# ---------------------------------------------------------
def process_ljk(image_np, key_answers, threshold_val=120):
    # 1. Ubah ke Grayscale & Blur
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Thresholding untuk memisahkan coretan/garis dari background
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Catatan: Dalam penerapan presisi penuh, disarankan melakukan Perspective Transform 
    # berdasarkan 4 titik sudut kertas/grid LJK.
    
    # Dimensi gambar
    h, w = gray.shape
    
    # Estimasi Area Grid Pilihan Ganda (Berdasarkan proporsi LJK terlampir)
    # Area PG berada di pertengahan atas lembar kertas
    grid_top = int(h * 0.26)
    grid_bottom = int(h * 0.55)
    grid_left = int(w * 0.05)
    grid_right = int(w * 0.95)
    
    roi_pg = thresh[grid_top:grid_bottom, grid_left:grid_right]
    roi_pg_color = image_np[grid_top:grid_bottom, grid_left:grid_right].copy()
    
    roi_h, roi_w = roi_pg.shape
    
    # 4 Kolom Pilihan Ganda (Soal 1-10, 11-20, 21-30, 31-40)
    col_width = roi_w / 4
    row_height = roi_h / 10
    
    detected_answers = {}
    
    # Peta Nomor Soal per Kolom
    col_ranges = [
        range(1, 11),   # Kolom 1: 1 - 10
        range(11, 21),  # Kolom 2: 11 - 20
        range(21, 31),  # Kolom 3: 21 - 30
        range(31, 41)   # Kolom 4: 31 - 40
    ]
    
    options = ['A', 'B', 'C', 'D']
    
    for c_idx, q_range in enumerate(col_ranges):
        col_x_start = c_idx * col_width
        
        for r_idx, q_num in enumerate(q_range):
            row_y_start = r_idx * row_height
            
            # Area opsi A, B, C, D berada di bagian kanan nomor soal (mengabaikan sel nomor)
            options_x_start = col_x_start + (col_width * 0.25)
            options_width = col_width * 0.75
            opt_cell_width = options_width / 4
            
            max_pixels = 0
            selected_option = "-"
            
            # Periksa setiap kotak A, B, C, D
            for opt_idx, opt_label in enumerate(options):
                x1 = int(options_x_start + (opt_idx * opt_cell_width) + (opt_cell_width * 0.15))
                x2 = int(options_x_start + ((opt_idx + 1) * opt_cell_width) - (opt_cell_width * 0.15))
                y1 = int(row_y_start + (row_height * 0.15))
                y2 = int(row_y_start + row_height - (row_height * 0.15))
                
                # Slicing cell
                cell = roi_pg[y1:y2, x1:x2]
                pixel_count = cv2.countNonZero(cell)
                
                # Deteksi coretan silang (piksel hitam/coretan lebih dominan dibanding cetakan huruf dasar)
                if pixel_count > threshold_val and pixel_count > max_pixels:
                    max_pixels = pixel_count
                    selected_option = opt_label
            
            detected_answers[q_num] = selected_option

    # 3. Hitung Skor & Evaluasi Jawaban
    score_correct = 0
    results = []
    
    for q_num in range(1, 41):
        user_ans = detected_answers.get(q_num, "-")
        key_ans = key_answers.get(q_num, "A")
        
        is_correct = (user_ans == key_ans)
        if is_correct:
            score_correct += 1
            
        results.append({
            "No Soal": q_num,
            "Jawaban Siswa": user_ans,
            "Kunci Jawaban": key_ans,
            "Status": "✅ Benar" if is_correct else "❌ Salah"
        })
        
    final_score = (score_correct / 40) * 100
    return final_score, score_correct, results, roi_pg_color

# ---------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------
sidebar = st.sidebar
sidebar.header("⚙️ Pengaturan & Kunci Jawaban")

# Form Input Kunci Jawaban
default_key = ["A", "B", "C", "D"] * 10
key_dict = {}

sidebar.subheader("Masukkan Kunci Jawaban (1 - 40)")
cols = sidebar.columns(2)
for i in range(1, 41):
    col_target = cols[0] if i <= 20 else cols[1]
    key_dict[i] = col_target.selectbox(
        f"Soal {i}", 
        options=['A', 'B', 'C', 'D'], 
        index=0, 
        key=f"kunci_{i}"
    )

threshold_sensitivity = sidebar.slider("Sensitivitas Coretan Silang", 50, 300, 120, 10)

# Input Upload Foto LJK
uploaded_file = st.file_uploader("Unggah Foto LJK (Format JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col_img, col_res = st.columns([1, 1])
    
    with col_img:
        st.subheader("🖼️ Lembar LJK Terunggah")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
    # Proses Koreksi
    score, correct_count, details, processed_roi = process_ljk(image, key_dict, threshold_sensitivity)
    
    with col_res:
        st.subheader("📊 Hasil Koreksi")
        st.metric(label="Nilai Akhir (Skala 100)", value=f"{score:.1f}")
        st.write(f"**Jumlah Benar:** {correct_count} / 40 Soal")
        
        # Tabel Rincian Jawaban
        df_results = pd.DataFrame(details)
        st.dataframe(df_results, height=400, use_container_width=True)

else:
    st.info("Silakan unggah foto lembar jawaban siswa untuk memulai koreksi.")
