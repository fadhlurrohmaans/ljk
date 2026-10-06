import io
import urllib.parse
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Scanner LJK Presisi - SMP YPI Pulogadung",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# FUNGSI DETEKSI ANCHOR DINAMIS (OPENCV AUTOMATIC OMR)
# ---------------------------------------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # Kiri-Atas
    rect[2] = pts[np.argmax(s)]       # Kanan-Bawah

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # Kanan-Atas
    rect[3] = pts[np.argmax(diff)]    # Kiri-Bawah
    return rect

def auto_detect_and_warp(img_np, target_w=800, target_h=1100):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    detected_pts = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Cari kontur segi empat terbesar yang menutupi area LJK (>20% luas gambar)
        if len(approx) == 4 and cv2.contourArea(c) > (img_np.shape[0] * img_np.shape[1] * 0.20):
            detected_pts = approx.reshape(4, 2)
            break

    if detected_pts is not None:
        rect = order_points(detected_pts)
        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_np, M, (target_w, target_h))
        return warped, True, rect
    
    # Fallback jika kontur luar tidak ditemukan
    return cv2.resize(img_np, (target_w, target_h)), False, None

def process_evalbee_grid(warped_img, key_answers, total_q=40, sensitivity_delta=15):
    h, w, _ = warped_img.shape
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

    # Grid dinamis yang terukur presisi setelah dikalibrasi warp
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
            if q_num > total_q:
                break

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
        if is_correct:
            score_correct += 1

        results.append({
            "No": q_num,
            "Siswa": user_ans,
            "Kunci": key_ans,
            "Status": "✅ Benar" if is_correct else ("❌ Salah" if user_ans != "-" else "⚪ Kosong")
        })

    final_score = (score_correct / float(total_q)) * 100.0
    return final_score, score_correct, results, annotated

# ---------------------------------------------------------
# SETUP STATE & KUNCI JAWABAN
# ---------------------------------------------------------
if 'num_questions' not in st.session_state:
    st.session_state['num_questions'] = 40

st.title("🎯 Pemindai LJK Otomatis SMP YPI")

with st.expander("⚙️ **Atur Kunci Jawaban & Sensitivitas**", expanded=False):
    num_questions = st.radio(
        "Jumlah Soal:", options=[40, 30],
        index=0 if st.session_state['num_questions'] == 40 else 1, horizontal=True
    )
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
# INPUT GAMBAR & AUTO DETEKSI ANCHOR
# ---------------------------------------------------------
tab_cam, tab_file = st.tabs(["📷 Kamera Langsung", "📁 Upload Foto Kertas"])
raw_image = None

with tab_cam:
    cam_file = st.camera_input("Foto Lembar LJK")
    if cam_file is not None:
        raw_image = Image.open(cam_file)

with tab_file:
    up_file = st.file_uploader("Upload Foto LJK dari Galeri", type=['jpg', 'jpeg', 'png'])
    if up_file is not None:
        raw_image = Image.open(up_file)

if raw_image is not None:
    try:
        raw_image = ImageOps.exif_transpose(raw_image)
    except Exception:
        pass

    img_np = np.array(raw_image.convert('RGB'))
    
    # Deteksi Otomatis & Alignment 4 Sudut LJK
    warped_img, is_detected, detected_corners = auto_detect_and_warp(img_np)

    if is_detected:
        st.success("✅ **Anchor LJK Terdeteksi Otomatis!** Gambar berhasil diluruskan secara presisi.")
    else:
        st.warning("⚠️ Batas luar LJK tidak terdeteksi utuh. Menggunakan mode penyesuaian standar. Pastikan latar belakang kertas kontras (misal: kertas putih di atas meja gelap).")

    score, correct_count, results, annotated_img = process_evalbee_grid(
        warped_img, key_dict, num_questions, delta_thresh
    )

    st.markdown("---")
    st.metric(label="📊 NILAI AKHIR", value=f"{score:.1f}")
    st.info(f"**Jawaban Benar:** {correct_count} dari {num_questions} Soal")

    st.subheader("🔍 Hasil Deteksi Bulatan Jawaban")
    st.image(annotated_img, use_container_width=True)

    st.subheader("📋 Rincian Jawaban Per Nomor")
    df_res = pd.DataFrame(results)
    st.dataframe(df_res, height=350, use_container_width=True)
