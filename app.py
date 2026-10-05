import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Scanner LJK Presisi - SMP YPI Pulogadung", layout="wide")

st.title("📱 Pemindai LJK Otomatis (Presisi Multi-Blok)")
st.caption("Khusus Format LJK SMP YPI Pulogadung (40 Soal Pilihan Ganda - 4 Kolom)")

# ---------------------------------------------------------
# DETEKSI 4 BLOK KOLOM TABEL LJK SECARA TERPISAH
# ---------------------------------------------------------
def detect_and_crop_4_columns(image_np):
    h, w, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Preprocessing: Blur & Adaptive Threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Batasi area pencarian di wilayah tengah foto (Pilihan Ganda: 22% - 60% tinggi)
    mask = np.zeros_like(thresh)
    mask[int(h * 0.22):int(h * 0.60), int(w * 0.03):int(w * 0.97)] = 255
    masked_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    contours, _ = cv2.findContours(masked_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect_ratio = bh / float(bw) if bw > 0 else 0
        area = bw * bh
        
        # Filter kontur yang mirip dengan bentuk 1 blok kolom (10 soal)
        if area > (w * h * 0.01) and 1.2 <= aspect_ratio <= 3.5:
            candidates.append((x, y, bw, bh))
            
    # Urutkan kontur dari kiri ke kanan berdasarkan koordinat X
    candidates = sorted(candidates, key=lambda b: b[0])
    
    column_crops = []
    
    # Jika berhasil menemukan setidaknya 4 blok kolom
    if len(candidates) >= 4:
        # Ambil 4 kontur terbaik dengan posisi Y yang sejajar
        selected_boxes = candidates[:4]
        for box in selected_boxes:
            x, y, bw, bh = box
            crop = image_np[y:y+bh, x:x+bw]
            resized = cv2.resize(crop, (250, 500))
            column_crops.append(resized)
        return column_crops, True
    else:
        # Fallback Matatis: Jika pencahayaan redup, bagi area Pilihan Ganda menjadi 4 bagian presisi
        roi_y1, roi_y2 = int(h * 0.27), int(h * 0.53)
        roi_x1, roi_x2 = int(w * 0.04), int(w * 0.96)
        
        roi_w = (roi_x2 - roi_x1) / 4.0
        for i in range(4):
            cx1 = int(roi_x1 + (i * roi_w))
            cx2 = int(roi_x1 + ((i + 1) * roi_w))
            crop = image_np[roi_y1:roi_y2, cx1:cx2]
            resized = cv2.resize(crop, (250, 500))
            column_crops.append(resized)
        return column_crops, False

# ---------------------------------------------------------
# EVALUASI JAWABAN DENGAN METODE RELATIVE DENSITY (SELISIH KEHITAMAN)
# ---------------------------------------------------------
def process_4_columns(column_crops, key_answers, sensitivity_delta=15):
    detected_answers = {}
    annotated_crops = []
    
    options = ['A', 'B', 'C', 'D']
    col_ranges = [
        range(1, 11),   # Kolom 1
        range(11, 21),  # Kolom 2
        range(21, 31),  # Kolom 3
        range(31, 41)   # Kolom 4
    ]
    
    for c_idx, q_range in enumerate(col_ranges):
        col_img = column_crops[c_idx].copy()
        gray = cv2.cvtColor(col_img, cv2.COLOR_RGB2GRAY)
        
        # Binerisasi untuk mengisolasi coretan pensil/pulpen
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        row_h = h / 10.0
        sub_col_w = w / 5.0  # Sub-kolom 0: No, Sub-kolom 1-4: A, B, C, D
        
        for r_idx, q_num in enumerate(q_range):
            row_y_start = r_idx * row_h
            
            densities = []
            cell_coords = []
            
            for opt_idx in range(4):
                # Koordinat sel A, B, C, D
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
            
            # Hitung rata-rata kehitaman sel lainnya pada baris yang sama
            other_vals = [v for i, v in enumerate(densities) if i != max_idx]
            avg_others = np.mean(other_vals) if len(other_vals) > 0 else 0
            
            selected_option = "-"
            # Jika sel tergelap memiliki selisih kehitaman signifikan di atas sel lainnya
            if (max_val - avg_others) > sensitivity_delta:
                selected_option = options[max_idx]
            
            detected_answers[q_num] = selected_option
            
            # Gambarkan indikator visual pada sel
            for opt_idx, (x1, y1, x2, y2) in enumerate(cell_coords):
                if opt_idx == max_idx and selected_option != "-":
                    color = (0, 255, 0)  # Hijau jika terdeteksi jawaban
                    cv2.rectangle(col_img, (x1, y1), (x2, y2), color, 2)
                else:
                    color = (200, 200, 200)  # Abu-abu untuk sel kosong
                    cv2.rectangle(col_img, (x1, y1), (x2, y2), color, 1)
                    
        annotated_crops.append(col_img)

    # Perhitungan Skor Akhir
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
            "Jawaban Siswa": user_ans,
            "Kunci Jawaban": key_ans,
            "Status": "✅ Benar" if is_correct else "❌ Salah"
        })
        
    final_score = (score_correct / 40.0) * 100.0
    return final_score, score_correct, results, annotated_crops

# ---------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------
st.sidebar.header("⚙️ Kunci Jawaban (40 Soal)")
key_dict = {}
cols = st.sidebar.columns(2)
for i in range(1, 41):
    col_target = cols[0] if i <= 20 else cols[1]
    key_dict[i] = col_target.selectbox(f"Soal {i}", ['A', 'B', 'C', 'D'], index=0, key=f"k_{i}")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Sensitivitas Deteksi Silang")
delta_thresh = st.sidebar.slider("Kontras Kehitaman Coretan (Delta)", 5, 50, 15, 1, 
                                 help="Kecilkan nilai jika coretan pensil tipis, naikkan jika terdeteksi ganda.")

uploaded_file = st.file_uploader("📷 Unggah / Ambil Foto LJK", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file)
        img_np = np.array(pil_image.convert('RGB'))
        
        # 1. Deteksi & Potong 4 Kolom Tabel
        col_crops, is_auto = detect_and_crop_4_columns(img_np)
        
        # 2. Evaluasi Kehitaman Relatif Jawaban
        score, correct_count, results, annotated_crops = process_4_columns(
            col_crops, key_dict, delta_thresh
        )
        
        if is_auto:
            st.success("✅ 4 Blok Tabel LJK berhasil terdeteksi dan dipisah secara presisi!")
        else:
            st.info("ℹ️ Menggunakan pemotongan grid 4 kolom standar.")

        # Tampilkan 4 Kolom Hasil Scan
        st.subheader("🔍 Visualisasi Pembacaan Per Kolom")
        c1, c2, c3, c4 = st.columns(4)
        c1.image(annotated_crops[0], caption="Soal 1 - 10", use_container_width=True)
        c2.image(annotated_crops[1], caption="Soal 11 - 20", use_container_width=True)
        c3.image(annotated_crops[2], caption="Soal 21 - 30", use_container_width=True)
        c4.image(annotated_crops[3], caption="Soal 31 - 40", use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Hasil Ringkasan Nilai")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric("Nilai Akhir", f"{score:.1f}")
            st.write(f"**Jumlah Benar:** {correct_count} dari 40 Soal")
            
        with res_col2:
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, height=300, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memproses gambar: {str(e)}")
