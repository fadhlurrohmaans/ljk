import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(page_title="Scanner LJK Presisi - SMP YPI Pulogadung", layout="wide")

st.title("📱 Pemindai LJK Otomatis SMP YPI Pulogadung")
st.caption("Khusus Format Pilihan Ganda (Fleksibel 30 atau 40 Soal)")

# ---------------------------------------------------------
# SIDEBAR: OPSI JUMLAH SOAL, KUNCI JAWABAN & SENSITIVITAS
# ---------------------------------------------------------
st.sidebar.header("📋 Mode Pengerjaan")
num_questions = st.sidebar.radio(
    "Pilih Jumlah Soal Pilihan Ganda:",
    options=[40, 30],
    index=0,
    help="Pilih 30 jika ujian hanya sampai Soal No. 30 (Kolom 1-3)."
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

# ---------------------------------------------------------
# FUNGSI OPTIMALISASI & DETEKSI 4 BINGKAI TABEL
# ---------------------------------------------------------
def prepare_image(file_bytes, max_dim=1200):
    raw_pil = Image.open(file_bytes)
    try:
        raw_pil = ImageOps.exif_transpose(raw_pil)
    except Exception:
        pass

    w, h = raw_pil.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        raw_pil = raw_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
    return np.array(raw_pil.convert('RGB'))

def find_and_crop_4_tables(image_np):
    h, w, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect_ratio = bh / float(bw) if bw > 0 else 0
        area = bw * bh
        
        if (w * h * 0.02) < area < (w * h * 0.25) and 1.2 <= aspect_ratio <= 3.8:
            boxes.append((x, y, bw, bh))
            
    boxes = sorted(boxes, key=lambda b: b[0])
    
    column_crops = []
    if len(boxes) >= 4:
        selected_boxes = boxes[:4]
        for box in selected_boxes:
            x, y, bw, bh = box
            crop = image_np[y:y+bh, x:x+bw]
            resized = cv2.resize(crop, (250, 500))
            column_crops.append(resized)
        return column_crops, True
    else:
        roi_y1, roi_y2 = int(h * 0.25), int(h * 0.55)
        roi_x1, roi_x2 = int(w * 0.03), int(w * 0.97)
        
        roi_w = (roi_x2 - roi_x1) / 4.0
        for i in range(4):
            cx1 = int(roi_x1 + (i * roi_w))
            cx2 = int(roi_x1 + ((i + 1) * roi_w))
            crop = image_np[roi_y1:roi_y2, cx1:cx2]
            resized = cv2.resize(crop, (250, 500))
            column_crops.append(resized)
        return column_crops, False

# ---------------------------------------------------------
# EVALUASI JAWABAN (DENGAN BATAS 30 ATAU 40 SOAL)
# ---------------------------------------------------------
def process_4_columns(column_crops, key_answers, total_q=40, sensitivity_delta=15):
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
        
        # Jika mode 30 soal, beri efek buram/terang pada kolom 4 (Soal 31-40)
        if q_range[0] > total_q:
            overlay = col_img.copy()
            cv2.rectangle(overlay, (0, 0), (col_img.shape[1], col_img.shape[0]), (230, 230, 230), -1)
            col_img = cv2.addWeighted(overlay, 0.6, col_img, 0.4, 0)
            cv2.putText(col_img, "TIDAK", (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(col_img, "DIGUNAKAN", (45, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            annotated_crops.append(col_img)
            continue

        gray = cv2.cvtColor(col_img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        row_h = h / 10.0
        sub_col_w = w / 5.0
        
        for r_idx, q_num in enumerate(q_range):
            if q_num > total_q:
                break
                
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
    
    for q_num in range(1, total_q + 1):
        user_ans = detected_answers.get(q_num, "-")
        key_ans = key_answers.get(q_num, "A")
        
        is_correct = (user_ans == key_ans)
        if is_correct:
            score_correct += 1
            
        results.append({
            "No": q_num,
            "Jawaban Siswa": user_ans,
            "Kunci Jawaban": key_ans,
            "Status": "✅ Benar" if is_correct else ("❌ Salah" if user_ans != "-" else "⚪ Kosong")
        })
        
    final_score = (score_correct / float(total_q)) * 100.0
    return final_score, score_correct, results, annotated_crops

# ---------------------------------------------------------
# INTERFACE UTAMA
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    f"📷 Unggah / Ambil Foto LJK SMP YPI ({num_questions} Soal)", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    try:
        img_np = prepare_image(uploaded_file, max_dim=1200)
        col_crops, is_auto = find_and_crop_4_tables(img_np)
        
        score, correct_count, results, annotated_crops = process_4_columns(
            col_crops, key_dict, num_questions, delta_thresh
        )
        
        if is_auto:
            st.success("✅ Bingkai LJK berhasil terisolasi secara otomatis!")
        else:
            st.info("ℹ️ Menggunakan pemotongan area standar LJK SMP YPI.")

        st.subheader("🔍 Visualisasi Pembacaan Per Kolom")
        c1, c2, c3, c4 = st.columns(4)
        c1.image(annotated_crops[0], caption="Soal 1 - 10", use_container_width=True)
        c2.image(annotated_crops[1], caption="Soal 11 - 20", use_container_width=True)
        c3.image(annotated_crops[2], caption="Soal 21 - 30", use_container_width=True)
        c4.image(annotated_crops[3], caption="Soal 31 - 40", use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"📊 Hasil Koreksi Otomatis (Total {num_questions} Soal)")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric("Nilai Akhir", f"{score:.1f}")
            st.write(f"**Jumlah Benar:** {correct_count} dari {num_questions} Soal")
            
        with res_col2:
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, height=300, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memproses gambar: {str(e)}")
