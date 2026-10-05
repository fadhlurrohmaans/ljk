import io
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(page_title="Scanner LJK Ringan - SMP YPI Pulogadung", layout="wide")

st.title("⚡ Pemindai LJK Otomatis & Super Ringan")
st.caption("Khusus Format LJK SMP YPI Pulogadung (40 Soal Pilihan Ganda - 4 Kolom)")

# ---------------------------------------------------------
# FUNGSI KOMPRESI FOTO SANGAT RINGAN
# ---------------------------------------------------------
def compress_and_load_image(file_bytes, max_dim=1000, quality=75):
    """
    Mengompres foto mentah dari HP (5-10 MB) menjadi file sangat kecil (~100-200 KB)
    dan menyesuaikan orientasi EXIF agar foto tidak terbalik.
    """
    raw_pil = Image.open(file_bytes)
    
    # Koreksi rotasi foto otomatis berdasarkan sensor HP (EXIF)
    try:
        raw_pil = ImageOps.exif_transpose(raw_pil)
    except Exception:
        pass

    # Resize dimensi foto
    w, h = raw_pil.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        raw_pil = raw_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Kompresi tingkat kualitas JPEG di dalam RAM
    buffer = io.BytesIO()
    raw_pil.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    
    compressed_pil = Image.open(buffer)
    size_kb = buffer.getbuffer().nbytes / 1024.0
    return np.array(compressed_pil), size_kb

# ---------------------------------------------------------
# DETEKSI 4 BLOK KOLOM TABEL LJK
# ---------------------------------------------------------
def detect_and_crop_4_columns(image_np):
    h, w, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    mask = np.zeros_like(thresh)
    mask[int(h * 0.22):int(h * 0.60), int(w * 0.03):int(w * 0.97)] = 255
    masked_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    contours, _ = cv2.findContours(masked_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect_ratio = bh / float(bw) if bw > 0 else 0
        area = bw * bh
        
        if area > (w * h * 0.01) and 1.2 <= aspect_ratio <= 3.5:
            candidates.append((x, y, bw, bh))
            
    candidates = sorted(candidates, key=lambda b: b[0])
    
    column_crops = []
    if len(candidates) >= 4:
        selected_boxes = candidates[:4]
        for box in selected_boxes:
            x, y, bw, bh = box
            crop = image_np[y:y+bh, x:x+bw]
            resized = cv2.resize(crop, (250, 500))
            column_crops.append(resized)
        return column_crops, True
    else:
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
# EVALUASI JAWABAN (RELATIVE DENSITY)
# ---------------------------------------------------------
def process_4_columns(column_crops, key_answers, sensitivity_delta=15):
    detected_answers = {}
    annotated_crops = []
    
    options = ['A', 'B', 'C', 'D']
    col_ranges = [
        range(1, 11),
        range(11, 21),
        range(21, 31),
        range(31, 41)
    ]
    
    for c_idx, q_range in enumerate(col_ranges):
        col_img = column_crops[c_idx].copy()
        gray = cv2.cvtColor(col_img, cv2.COLOR_RGB2GRAY)
        
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        row_h = h / 10.0
        sub_col_w = w / 5.0
        
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
                    cv2.rectangle(col_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(col_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
                    
        annotated_crops.append(col_img)

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
delta_thresh = st.sidebar.slider("Kontras Kehitaman Coretan (Delta)", 5, 50, 15, 1)

# Pengunggah File Utama yang Memicu Kamera Asli HP
uploaded_file = st.file_uploader(
    "📷 Ambil Foto lewat Kamera HP / Unggah Gambar LJK", 
    type=['jpg', 'jpeg', 'png'],
    help="Saat ditekan dari HP, pilih menu Kamera untuk mengambil foto langsung."
)

if uploaded_file is not None:
    try:
        # 1. Kompresi Foto Ekstrem & Perbaikan EXIF Rotasi
        img_np, compressed_size_kb = compress_and_load_image(uploaded_file, max_dim=1000, quality=75)
        
        # 2. Deteksi & Potong 4 Kolom
        col_crops, is_auto = detect_and_crop_4_columns(img_np)
        
        # 3. Evaluasi Jawaban
        score, correct_count, results, annotated_crops = process_4_columns(
            col_crops, key_dict, delta_thresh
        )
        
        st.success(f"⚡ Foto berhasil dikompresi menjadi {compressed_size_kb:.1f} KB & diproses secara instan!")

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
else:
    st.info("Tekan tombol di atas. Pada HP, pilih 'Kamera' untuk memotret LJK secara langsung.")
