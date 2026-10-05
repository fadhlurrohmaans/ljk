import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

st.set_page_config(page_title="Koreksi LJK - SMP YPI Pulogadung", layout="wide")

st.title("📋 Pemindai & Koreksi LJK Otomatis")
st.caption("Khusus Format LJK SMP YPI Pulogadung (40 Soal Pilihan Ganda - Model Silang)")

def process_ljk(pil_img, key_answers, threshold_val=100, min_pixels=60, 
                top_pct=28, bottom_pct=49, left_pct=4, right_pct=96,
                shift_x=0, shift_y=0):
    
    gray = ImageOps.grayscale(pil_img)
    img_np = np.array(gray)
    
    h, w = img_np.shape
    
    # Hitung batas grid dengan pergeseran (shift_x dan shift_y)
    grid_top = int(h * (top_pct / 100.0)) + shift_y
    grid_bottom = int(h * (bottom_pct / 100.0)) + shift_y
    grid_left = int(w * (left_pct / 100.0)) + shift_x
    grid_right = int(w * (right_pct / 100.0)) + shift_x
    
    # Kunci koordinat agar tidak melebihi batas gambar
    grid_top = max(0, min(h, grid_top))
    grid_bottom = max(0, min(h, grid_bottom))
    grid_left = max(0, min(w, grid_left))
    grid_right = max(0, min(w, grid_right))
    
    roi = img_np[grid_top:grid_bottom, grid_left:grid_right]
    roi_h, roi_w = roi.shape
    
    if roi_h == 0 or roi_w == 0:
        return 0, 0, [], pil_img

    binary_roi = (roi < threshold_val).astype(np.uint8)

    col_width = roi_w / 4.0
    row_height = roi_h / 10.0
    
    detected_answers = {}
    col_ranges = [
        range(1, 11),   # Soal 1-10
        range(11, 21),  # Soal 11-20
        range(21, 31),  # Soal 21-30
        range(31, 41)   # Soal 31-40
    ]
    options = ['A', 'B', 'C', 'D']
    
    debug_img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(debug_img)
    
    # Gambar garis batas luar grid (Warna Biru)
    draw.rectangle([grid_left, grid_top, grid_right, grid_bottom], outline="blue", width=2)
    
    for c_idx, q_range in enumerate(col_ranges):
        col_x_start = c_idx * col_width
        
        for r_idx, q_num in enumerate(q_range):
            row_y_start = r_idx * row_height
            sub_col_width = col_width / 5.0
            
            max_pixels = 0
            selected_option = "-"
            
            for opt_idx, opt_label in enumerate(options):
                opt_x_start = col_x_start + ((opt_idx + 1) * sub_col_width)
                
                x1 = int(opt_x_start + (sub_col_width * 0.20))
                x2 = int(opt_x_start + (sub_col_width * 0.80))
                y1 = int(row_y_start + (row_height * 0.20))
                y2 = int(row_y_start + (row_height * 0.80))
                
                x1, x2 = max(0, x1), min(roi_w, x2)
                y1, y2 = max(0, y1), min(roi_h, y2)
                
                cell = binary_roi[y1:y2, x1:x2]
                pixel_count = np.sum(cell) if cell.size > 0 else 0
                
                abs_x1 = grid_left + x1
                abs_y1 = grid_top + y1
                abs_x2 = grid_left + x2
                abs_y2 = grid_top + y2
                
                if pixel_count > min_pixels and pixel_count > max_pixels:
                    max_pixels = pixel_count
                    selected_option = opt_label
                
                # Gambar kotak hijau untuk area pemeriksaan
                draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline="green", width=1)
            
            detected_answers[q_num] = selected_option

    # Perhitungan Skor
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
    return final_score, score_correct, results, debug_img

# Sidebar Kunci Jawaban
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

st.sidebar.markdown("---")
st.sidebar.header("🎯 Alignment & Geser Grid")

# Kontrol Pergeseran Gambar
shift_x = st.sidebar.slider("⬅️ Geser Kiri / Kanan ➡️ (px)", -200, 200, 0, 5)
shift_y = st.sidebar.slider("⬆️ Geser Atas / Bawah ⬇️ (px)", -200, 200, 0, 5)

with st.sidebar.expander("📐 Ukuran & Margin Grid"):
    top_crop = st.slider("Posisi Atas (%)", 10, 40, 28)
    bottom_crop = st.slider("Posisi Bawah (%)", 40, 70, 49)
    left_crop = st.slider("Margin Kiri (%)", 0, 20, 4)
    right_crop = st.slider("Margin Kanan (%)", 80, 100, 96)

with st.sidebar.expander("🔍 Sensitivitas Coretan"):
    threshold_sensitivity = st.slider("Kehitaman Pensil (Threshold)", 50, 200, 100, 5)
    min_pixel_cutoff = st.slider("Minimal Piksel Coretan", 20, 200, 60, 5)

# Upload Foto LJK
uploaded_file = st.file_uploader("Unggah Foto Lembar Jawaban LJK (JPG / PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        score, correct_count, details, debug_image = process_ljk(
            image, key_dict, threshold_sensitivity, min_pixel_cutoff, 
            top_crop, bottom_crop, left_crop, right_crop,
            shift_x, shift_y
        )
        
        col_img, col_res = st.columns([1, 1])
        
        with col_img:
            st.subheader("🖼️ Area Pembacaan Sistem")
            st.image(debug_image, use_container_width=True, caption="Garis biru: Batas Luar Grid. Kotak hijau: Sel Pilihan Ganda.")
            
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
