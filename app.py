import io
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Scanner LJK Otomatis - SMP YPI Pulogadung",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# FUNGSI 1: ALIGNMENT BERBASIS TEMPLATE
# ---------------------------------------------------------
def align_image_to_template(img_siswa_np, img_template_np, max_features=10000, keep_percent=0.2):
    gray_siswa = cv2.cvtColor(img_siswa_np, cv2.COLOR_RGB2GRAY)
    gray_template = cv2.cvtColor(img_template_np, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(max_features)
    kps_siswa, des_siswa = orb.detectAndCompute(gray_siswa, None)
    kps_template, des_template = orb.detectAndCompute(gray_template, None)

    if des_siswa is None or des_template is None:
        return img_siswa_np, False

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des_siswa, des_template)
    matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    if len(matches) < 10: return img_siswa_np, False

    pts_siswa = np.zeros((len(matches), 2), dtype="float32")
    pts_template = np.zeros((len(matches), 2), dtype="float32")

    for i, m in enumerate(matches):
        pts_siswa[i] = kps_siswa[m.queryIdx].pt
        pts_template[i] = kps_template[m.trainIdx].pt

    H, mask = cv2.findHomography(pts_siswa, pts_template, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if H is not None:
        h, w = img_template_np.shape[:2]
        aligned_img = cv2.warpPerspective(img_siswa_np, H, (w, h))
        return aligned_img, True
    
    return img_siswa_np, False

# ---------------------------------------------------------
# FUNGSI 2: DETEKSI TABEL OTOMATIS (CONTOUR DETECTION)
# ---------------------------------------------------------
def process_auto_grid(warped_img, key_answers, total_q=40, sensitivity_delta=15):
    annotated = warped_img.copy()
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    
    # Binarisasi untuk mencari garis tabel
    thresh_val, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Operasi Morfologi untuk mempertebal garis tabel vertikal & horizontal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Cari kontur tabel
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter kontur yang menyerupai tabel Pilihan Ganda (Asumsi 4 kolom)
    table_contours = []
    h_img, w_img = gray.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = h / float(w)
        # Tabel soal PG memiliki ciri: tinggi > lebar, dan areanya cukup besar
        if 1.5 < aspect_ratio < 4.0 and w > (w_img * 0.1) and h > (h_img * 0.15):
            table_contours.append((x, y, w, h))
            
    # Urutkan tabel dari kiri ke kanan (Kolom 1-10, 11-20, dst)
    table_contours = sorted(table_contours, key=lambda b: b[0])
    
    detected_answers = {}
    options = ['A', 'B', 'C', 'D']
    
    # Jika gagal mendeteksi 4 tabel utama, sistem butuh foto yang lebih kontras
    if len(table_contours) == 0:
        return 0, 0, [], annotated, False 

    # Ambil maksimal 4 tabel sesuai jumlah soal standar
    table_contours = table_contours[:4]
    
    q_ranges = [range(1, 11), range(11, 21), range(21, 31), range(31, 41)]
    
    for idx, (tx, ty, tw, th) in enumerate(table_contours):
        if idx >= len(q_ranges): break
        current_range = q_ranges[idx]
        
        cv2.rectangle(annotated, (tx, ty), (tx+tw, ty+th), (0, 0, 255), 2) # Gambar batas tabel
        
        row_height = th / 10.0
        col_width = tw / 5.0 # Dibagi 5: [Nomor Soal, A, B, C, D]
        
        for r_idx, q_num in enumerate(current_range):
            if q_num > total_q: break
            
            row_y_start = ty + int(r_idx * row_height)
            densities = []
            cell_coords = []
            
            for opt_idx in range(4):
                # opt_idx + 1 untuk melewati sel Nomor Soal
                cx1 = tx + int((opt_idx + 1) * col_width) + int(col_width * 0.15)
                cx2 = tx + int((opt_idx + 2) * col_width) - int(col_width * 0.15)
                cy1 = row_y_start + int(row_height * 0.15)
                cy2 = row_y_start + int(row_height * 0.85)
                
                cell_coords.append((cx1, cy1, cx2, cy2))
                
                # Ekstrak area jawaban dan hitung kehitaman (pixel density)
                cell = binary[cy1:cy2, cx1:cx2]
                pixel_count = cv2.countNonZero(cell) if cell.size > 0 else 0
                densities.append(pixel_count)
                
            # Logika Pemilihan Jawaban
            max_val = max(densities)
            max_idx = densities.index(max_val)
            other_vals = [v for i, v in enumerate(densities) if i != max_idx]
            avg_others = np.mean(other_vals) if len(other_vals) > 0 else 0

            selected_option = "-"
            # Jika selisih hitam antara jawaban tergelap dengan rata-rata jawaban lain cukup besar
            if (max_val - avg_others) > (sensitivity_delta * 10): 
                selected_option = options[max_idx]

            detected_answers[q_num] = selected_option

            # Gambar visualisasi hasil deteksi
            for opt_idx, (cx1, cy1, cx2, cy2) in enumerate(cell_coords):
                if opt_idx == max_idx and selected_option != "-":
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2) # Hijau jika dijawab
                else:
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255, 200, 0), 1) # Kuning u/ area kosong

    score_correct = 0
    results = []
    for q_num in range(1, total_q + 1):
        user_ans = detected_answers.get(q_num, "-")
        key_ans = key_answers.get(q_num, "A")
        is_correct = (user_ans == key_ans)
        if is_correct: score_correct += 1

        results.append({
            "No": q_num, "Siswa": user_ans, "Kunci": key_ans,
            "Status": "✅ Benar" if is_correct else ("❌ Salah" if user_ans != "-" else "⚪ Kosong")
        })

    final_score = (score_correct / float(total_q)) * 100.0
    return final_score, score_correct, results, annotated, True


# ---------------------------------------------------------
# SETUP STATE UI & PENGATURAN
# ---------------------------------------------------------
st.title("🎯 Pemindai LJK (Auto-Detect Grid)")

if 'num_questions' not in st.session_state: st.session_state['num_questions'] = 40

with st.expander("⚙️ **Atur Kunci Jawaban & Sensitivitas**", expanded=False):
    num_questions = st.radio("Jumlah Soal:", options=[40, 30], index=0 if st.session_state['num_questions'] == 40 else 1, horizontal=True)
    st.session_state['num_questions'] = num_questions

    if 'key_answers_list' not in st.session_state or len(st.session_state['key_answers_list']) != num_questions:
        st.session_state['key_answers_list'] = ['A'] * num_questions

    quick_string = "".join(st.session_state['key_answers_list'])
    user_input = st.text_input(f"Ketik {num_questions} Kunci (Contoh: ABCD...):", value=quick_string).upper()
    cleaned_keys = [char for char in user_input if char in ['A', 'B', 'C', 'D']]
    if len(cleaned_keys) == num_questions:
        st.session_state['key_answers_list'] = cleaned_keys

    # Skala slider diperbesar karena menggunakan mode deteksi contour area
    delta_thresh = st.slider("Sensitivitas Kehitaman Pensil", 10, 100, 30, 5)

num_questions = st.session_state['num_questions']
key_dict = {i + 1: st.session_state['key_answers_list'][i] for i in range(num_questions)}

# ---------------------------------------------------------
# UPLOAD TEMPLATE MASTER & LJK SISWA
# ---------------------------------------------------------
st.info("💡 **Mode Otomatis Aktif:** Sistem akan melacak garis bingkai tabel pada area Pilihan Ganda secara mandiri tanpa slider kalibrasi.")
col1, col2 = st.columns(2)

with col1:
    template_file = st.file_uploader("Upload Master (Kosong)", type=['jpg', 'jpeg', 'png'], key="template")
with col2:
    siswa_file = st.file_uploader("Upload LJK Jawaban", type=['jpg', 'jpeg', 'png'], key="siswa")

if template_file is not None and siswa_file is not None:
    try:
        raw_template = Image.open(template_file)
        raw_template = ImageOps.exif_transpose(raw_template)
        raw_template = raw_template.resize((900, 1200))
        img_template_np = np.array(raw_template.convert('RGB'))

        raw_siswa = Image.open(siswa_file)
        raw_siswa = ImageOps.exif_transpose(raw_siswa)
        img_siswa_np = np.array(raw_siswa.convert('RGB'))

        with st.spinner("Mencocokkan pola LJK..."):
            aligned_img, is_aligned = align_image_to_template(img_siswa_np, img_template_np)

        if is_aligned:
            st.success("✅ Pola LJK berhasil diluruskan!")
            
            # Eksekusi Pembacaan Jawaban Otomatis
            score, correct_count, results, annotated_img, auto_success = process_auto_grid(
                aligned_img, key_dict, num_questions, delta_thresh
            )

            if auto_success:
                st.markdown("---")
                st.metric(label="📊 NILAI AKHIR", value=f"{score:.1f}")
                st.info(f"**Jawaban Benar:** {correct_count} dari {num_questions} Soal")

                st.subheader("🔍 Hasil Deteksi Otomatis")
                st.caption("Garis Merah: Tabel Terdeteksi Otomatis | Kotak Hijau/Kuning: Anchor Jawaban")
                st.image(annotated_img, use_container_width=True)

                st.subheader("📋 Rincian Jawaban Per Nomor")
                df_res = pd.DataFrame(results)
                st.dataframe(df_res, height=350, use_container_width=True)
            else:
                st.error("❌ Gagal mendeteksi bingkai tabel. Pastikan foto jelas, terang, dan batas tabel Pilihan Ganda tidak terpotong.")
                st.image(annotated_img, caption="Hasil Pencarian Tabel", width=400)
        else:
            st.error("❌ Gagal mencocokkan kemiringan LJK dengan Template Master.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
