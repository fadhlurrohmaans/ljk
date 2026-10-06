import io
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Scanner LJK ORB - SMP YPI Pulogadung",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# FUNGSI 1: ALIGNMENT BERBASIS TEMPLATE (ORB + HOMOGRAPHY)
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

    if len(matches) < 10:
        return img_siswa_np, False

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
# FUNGSI 2: PEMROSESAN GRID JAWABAN DENGAN KALIBRASI DINAMIS
# ---------------------------------------------------------
def process_evalbee_grid(warped_img, key_answers, config, total_q=40, sensitivity_delta=15):
    h, w, _ = warped_img.shape
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

    # Menggunakan nilai dari kalibrasi UI
    y1_global = int(h * config['y_start'])
    y2_global = int(h * config['y_end'])
    row_h = (y2_global - y1_global) / 10.0

    # Titik X awal untuk setiap kolom
    col_starts = [config['x_col1'], config['x_col2'], config['x_col3'], config['x_col4']]
    col_width = config['col_width']
    
    col_ranges = [range(1, 11), range(11, 21), range(21, 31), range(31, 41)]
    options = ['A', 'B', 'C', 'D']
    detected_answers = {}
    annotated = warped_img.copy()

    for c_idx, q_range in enumerate(col_ranges):
        x1_col = int(w * col_starts[c_idx])
        x2_col = int(w * (col_starts[c_idx] + col_width))
        sub_col_w = (x2_col - x1_col) / 5.0 # Dibagi 5 karena ada kolom nomor soal

        if q_range[0] > total_q:
            cv2.rectangle(annotated, (x1_col, y1_global), (x2_col, y2_global), (200, 200, 200), -1)
            continue

        for r_idx, q_num in enumerate(q_range):
            if q_num > total_q: break

            row_y1 = int(y1_global + (r_idx * row_h))
            densities = []
            cell_coords = []

            for opt_idx in range(4):
                # opt_idx + 1 untuk melewati kolom nomor soal
                cx1 = int(x1_col + ((opt_idx + 1) * sub_col_w) + (sub_col_w * 0.1))
                cx2 = int(x1_col + ((opt_idx + 2) * sub_col_w) - (sub_col_w * 0.1))
                cy1 = int(row_y1 + (row_h * 0.15))
                cy2 = int(row_y1 + row_h - (row_h * 0.15))

                cell_coords.append((cx1, cy1, cx2, cy2))
                cell = binary[cy1:cy2, cx1:cx2]
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

            # Menggambar kotak deteksi
            for opt_idx, (cx1, cy1, cx2, cy2) in enumerate(cell_coords):
                if opt_idx == max_idx and selected_option != "-":
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)

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
    return final_score, score_correct, results, annotated


# ---------------------------------------------------------
# SETUP STATE UI & PENGATURAN
# ---------------------------------------------------------
st.title("🎯 Pemindai LJK Otomatis (Metode Template)")

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

    delta_thresh = st.slider("Sensitivitas Kehitaman Pensil", 5, 50, 15, 1)

num_questions = st.session_state['num_questions']
key_dict = {i + 1: st.session_state['key_answers_list'][i] for i in range(num_questions)}

# --- PANEL KALIBRASI GRID ---
with st.expander("🔧 **Kalibrasi Posisi Grid PG (Geser jika kotak merah tidak pas)**", expanded=True):
    st.caption("Sesuaikan slider di bawah ini agar kotak deteksi pas berada di atas huruf opsi jawaban.")
    col_a, col_b = st.columns(2)
    with col_a:
        y_start = st.slider("Batas Atas Grid (Y Start)", 0.20, 0.40, 0.290, 0.005)
        y_end = st.slider("Batas Bawah Grid (Y End)", 0.40, 0.65, 0.510, 0.005)
        col_width = st.slider("Lebar Tabel Per Kolom", 0.10, 0.30, 0.180, 0.005)
    with col_b:
        x_col1 = st.slider("Posisi Kolom 1 (No 1-10)", 0.00, 0.20, 0.050, 0.005)
        x_col2 = st.slider("Posisi Kolom 2 (No 11-20)", 0.20, 0.40, 0.280, 0.005)
        x_col3 = st.slider("Posisi Kolom 3 (No 21-30)", 0.40, 0.60, 0.515, 0.005)
        x_col4 = st.slider("Posisi Kolom 4 (No 31-40)", 0.60, 0.85, 0.750, 0.005)

grid_config = {
    'y_start': y_start, 'y_end': y_end, 'col_width': col_width,
    'x_col1': x_col1, 'x_col2': x_col2, 'x_col3': x_col3, 'x_col4': x_col4
}

# ---------------------------------------------------------
# UPLOAD TEMPLATE MASTER & LJK SISWA
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Upload Template Master LJK")
    template_file = st.file_uploader("Upload Master", type=['jpg', 'jpeg', 'png'], key="template")

with col2:
    st.markdown("### 2. Upload LJK Siswa")
    siswa_file = st.file_uploader("Upload LJK Jawaban", type=['jpg', 'jpeg', 'png'], key="siswa")

if template_file is not None and siswa_file is not None:
    try:
        raw_template = Image.open(template_file)
        raw_template = ImageOps.exif_transpose(raw_template)
        raw_template = raw_template.resize((800, 1100))
        img_template_np = np.array(raw_template.convert('RGB'))

        raw_siswa = Image.open(siswa_file)
        raw_siswa = ImageOps.exif_transpose(raw_siswa)
        img_siswa_np = np.array(raw_siswa.convert('RGB'))

        with st.spinner("Mencocokkan pola LJK..."):
            aligned_img, is_aligned = align_image_to_template(img_siswa_np, img_template_np)

        if is_aligned:
            st.success("✅ Pola LJK Siswa berhasil disamakan dengan Template Master!")
            
            # Eksekusi Pembacaan Jawaban dengan Konfigurasi Grid Baru
            score, correct_count, results, annotated_img = process_evalbee_grid(
                aligned_img, key_dict, grid_config, num_questions, delta_thresh
            )

            st.markdown("---")
            st.metric(label="📊 NILAI AKHIR", value=f"{score:.1f}")
            st.info(f"**Jawaban Benar:** {correct_count} dari {num_questions} Soal")

            st.subheader("🔍 Hasil Deteksi (Cek Posisi Kotak Di Sini)")
            st.caption("Jika kotak merah/hijau tidak pas di opsi jawaban, geser slider di menu 'Kalibrasi Posisi Grid PG' di atas.")
            st.image(annotated_img, use_container_width=True)

            st.subheader("📋 Rincian Jawaban Per Nomor")
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, height=350, use_container_width=True)
        else:
            st.error("❌ Gagal mencocokkan LJK. Pastikan foto siswa tidak terlalu blur dan format kertasnya persis dengan Master Template.")
            st.image(img_siswa_np, caption="Foto LJK Siswa (Gagal Diproses)", width=400)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
elif template_file is None and siswa_file is not None:
    st.warning("⚠️ Harap upload **Template Master LJK** terlebih dahulu di kolom sebelah kiri!")
