import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Koreksi LJK - SMP YPI Pulogadung", layout="wide")

st.title("📋 Aplikasi Koreksi LJK Otomatis")
st.caption("Khusus Format LJK SMP YPI Pulogadung (40 Soal Pilihan Ganda - Model Silang)")

# 2. Fungsi Pemrosesan Gambar LJK
def process_ljk(pil_img, key_answers, threshold_val=110):
    # Ubah gambar ke skala abu-abu (grayscale)
    gray = ImageOps.grayscale(pil_img)
    img_np = np.array(gray)
    
    h, w = img_np.shape
    
    # Area Grid Pilihan Ganda (25% - 55% dari tinggi kertas)
    grid_top = int(h * 0.25)
    grid_bottom = int(h * 0.55)
    grid_left = int(w * 0.05)
    grid_right = int(w * 0.95)
    
    roi = img_np[grid_top:grid_bottom, grid_left:grid_right]
    roi_h, roi_w = roi.shape
    
    if roi_h == 0 or roi_w == 0:
        return 0, 0, []

    # Binerisasi: Piksel di bawah nilai threshold diidentifikasi sebagai coretan/pensil
    binary_roi = (roi < threshold_val).astype(np.uint8)

    # 4 Kolom x 10 Baris Soal
    col_width = roi_w / 4.0
    row_height = roi_h / 10.0
    
    detected_answers = {}
    col_ranges = [
        range(1, 11),   # Soal 1 - 10
        range(11, 21),  # Soal 11 - 20
        range(21, 31),  # Soal 21 - 30
        range(31, 41)   # Soal 31 - 40
    ]
    options = ['A', 'B', 'C', 'D']
    
    for c_idx, q_range in enumerate(col_ranges):
        col_x_start = c_idx * col_width
        
        for r_idx, q_num in enumerate(q_range):
            row_y_start = r_idx * row_height
            options_x_start = col_x_start + (col_width * 0.25)
            options_width = col_width * 0.75
            opt_cell_width = options_width / 4.0
            
            max_pixels = 0
            selected_option = "-"
            
            for opt_idx, opt_label in enumerate(options):
                x1 = int(options_x_start + (opt_idx * opt_cell_width) + (opt_cell_width * 0.15))
                x2 = int(options_x_start + ((opt_idx + 1) * opt_cell_width) - (opt_cell_width * 0.15))
                y1 = int(row_y_start + (row_height * 0.15))
                y2 = int(row_y_start + row_height - (row_height * 0.15))
                
                # Batasi koordinat agar tidak keluar dari gambar
                x1, x2 = max(0, x1), min(roi_w, x2)
                y1, y2 = max(0, y1), min(roi_h, y2)
                
                cell = binary_roi[y1:y2, x1:x2]
                
                if cell.size > 0:
                    pixel_count = np.sum(cell)
                    # Jika jumlah piksel hitam memenuhi syarat, catat opsi tersebut
                    if pixel_count > 20 and pixel_count > max_pixels:
                        max_pixels = pixel_count
                        selected_option = opt_label
            
            detected_answers[q_num] = selected_option

    # 3. Perhitungan Skor
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
        
    final_score = (score_correct / 40.0) * 100.0
    return final_score, score_correct, results

# 3. Antarmuka Pengguna (UI) Sidebar & Form
st.sidebar.header("⚙️ Pengaturan Kunci Jawaban")

key_dict = {}
cols = st.sidebar.columns(2)
for i in range(1, 41):
    col_target = cols[0] if i <= 20 else cols[1]
    key_dict[i] = col_target.selectbox(
        f"Soal {i}", 
        options=['A', 'B', 'C', 'D'], 
        index=0, 
        key=f"kunci_{i}"
    )

threshold_sensitivity = st.sidebar.slider("Sensitivitas Deteksi Silang", 50, 200, 110, 5)

# 4. Upload Foto LJK & Tampilkan Hasil
uploaded_file = st.file_uploader("Unggah Foto Lembar Jawaban LJK (JPG / PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        col_img, col_res = st.columns([1, 1])
        
        with col_img:
            st.subheader("🖼️ Lembar LJK Terunggah")
            st.image(image, use_container_width=True)
            
        score, correct_count, details = process_ljk(image, key_dict, threshold_sensitivity)
        
        with col_res:
            st.subheader("📊 Hasil Koreksi")
            st.metric(label="Nilai Akhir (Skala 100)", value=f"{score:.1f}")
            st.write(f"**Jumlah Benar:** {correct_count} dari 40 Soal")
            
            df_results = pd.DataFrame(details)
            st.dataframe(df_results, height=400, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memproses gambar: {str(e)}")
else:
    st.info("Silakan unggah foto LJK siswa untuk memulai koreksi otomatis.")
