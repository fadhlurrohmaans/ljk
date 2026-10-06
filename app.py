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

    # Inisialisasi ORB (Pendeteksi Pola)
    orb = cv2.ORB_create(max_features)
    kps_siswa, des_siswa = orb.detectAndCompute(gray_siswa, None)
    kps_template, des_template = orb.detectAndCompute(gray_template, None)

    if des_siswa is None or des_template is None:
        return img_siswa_np, False

    # Pencocokan titik unik
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des_siswa, des_template)
    matches = sorted(matches, key=lambda x: x.distance)

    # Ambil persentase kecocokan terbaik
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    if len(matches) < 10:
        return img_siswa_np, False

    pts_siswa = np.zeros((len(matches), 2), dtype="float32")
    pts_template = np.zeros((len(matches), 2), dtype="float32")

    for i, m in enumerate(matches):
        pts_siswa[i] = kps_siswa[m.queryIdx].pt
        pts_template[i] = kps_template[m.trainIdx].pt

    # Hitung matriks Homography
    H, mask = cv2.findHomography(pts_siswa, pts_template, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if H is not None:
        # Luruskan LJK Siswa agar ukurannya 100% sama dengan Template
        h, w = img_template_np.shape[:2]
        aligned_img = cv2.warpPerspective(img_siswa_np, H, (w, h))
        return aligned_img, True
    
    return img_siswa_np, False

# ---------------------------------------------------------
# FUNGSI 2: PEMROSESAN GRID JAWABAN
# ---------------------------------------------------------
def process_evalbee_grid(warped_img, key_answers, total_q=40, sensitivity_delta=15):
    h, w, _ = warped_img.shape
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

    # Area Pilihan Ganda berdasarkan proporsi Template
    y1_global = int(h * 0.28)
    y2_global = int(h * 0.58)
    row_h = (y2_global - y1_global) / 10.0

    col_x_pcts = [(0.04, 0.25), (0.27, 0.48), (0.50, 0.71), (0.73, 0.94)]
    col_ranges = [range(1, 11), range(11, 21), range(21, 31), range(31, 41)]
    options = ['A', 'B', 'C', 'D']
    detected_answers = {}
    annotated = warped_img.copy()

    for c_idx, q_range in enumerate(col_ranges):
        x_start_pct, x_end_pct = col_x_pcts[c_idx]
        x1_col = int(w * x_start_pct)
        x2_col = int(w * x_end_pct)
        col_w = x2_col - x1_col
        sub_col_w = col_w / 5.0

        if q_range[0] > total_q:
            cv2.rectangle(annotated, (x1_col, y1_global), (x2_col, y2_global), (200, 200, 200), -1)
            continue

        for r_idx, q_num in enumerate(q_range):
            if q_num > total_q: break

            row_y1 = int(y1_global + (r_idx * row_h))
            densities = []
            cell_coords = []

            for opt_idx in range(4):
                cx1 = int(x1_col + ((opt_idx + 1) * sub_col_w) + (sub_col_w * 0.12))
                cx2 = int(x1_col + ((opt_idx + 2) * sub_col_w) - (sub_col_w * 0.12))
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

            for opt_idx, (cx1, cy1, cx2, cy2) in enumerate(cell_coords):
                if opt_idx == max_idx and selected_option != "-":
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (220, 220, 220), 1)

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

# ---------------------------------------------------------
# UPLOAD TEMPLATE MASTER & LJK SISWA
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Upload Template Master LJK")
    st.caption("Gunakan foto LJK Kosong yang sangat lurus sebagai acuan patokan.")
    template_file = st.file_uploader("Upload Master", type=['jpg', 'jpeg', 'png'], key="template")

with col2:
    st.markdown("### 2. Upload LJK Siswa")
    st.caption("Pastikan keseluruhan kertas LJK dan teks terekam di dalam foto.")
    siswa_file = st.file_uploader("Upload LJK Jawaban", type=['jpg', 'jpeg', 'png'], key="siswa")

if template_file is not None and siswa_file is not None:
    try:
        # Load & Transpose Template
        raw_template = Image.open(template_file)
        raw_template = ImageOps.exif_transpose(raw_template)
        # Standarisasi ukuran master ke resolusi ideal
        raw_template = raw_template.resize((800, 1100))
        img_template_np = np.array(raw_template.convert('RGB'))

        # Load & Transpose Siswa
        raw_siswa = Image.open(siswa_file)
        raw_siswa = ImageOps.exif_transpose(raw_siswa)
        img_siswa_np = np.array(raw_siswa.convert('RGB'))

        # PROSES ALIGNMENT (Menyamakan perspektif siswa dengan master)
        with st.spinner("Mencocokkan pola LJK..."):
            aligned_img, is_aligned = align_image_to_template(img_siswa_np, img_template_np)

        if is_aligned:
            st.success("✅ Pola LJK Siswa berhasil disamakan dengan Template Master!")
            
            # Eksekusi Pembacaan Jawaban
            score, correct_count, results, annotated_img = process_evalbee_grid(
                aligned_img, key_dict, num_questions, delta_thresh
            )

            st.markdown("---")
            st.metric(label="📊 NILAI AKHIR", value=f"{score:.1f}")
            st.info(f"**Jawaban Benar:** {correct_count} dari {num_questions} Soal")

            st.subheader("🔍 Hasil Deteksi (Sudah Diluruskan)")
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
