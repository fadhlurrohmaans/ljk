import io
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Scanner LJK Presisi - SMP YPI Pulogadung",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Pemindai LJK Presisi (Metode EvalBee)")
st.caption("Pindai Instan Kamera Native HP + Auto Warp Perspective Alignment - SMP YPI Pulogadung")

# ---------------------------------------------------------
# SIDEBAR: KONTROL JUMLAH SOAL, KUNCI, DAN SENSITIVITAS
# ---------------------------------------------------------
st.sidebar.header("📋 Mode Pengerjaan")
num_questions = st.sidebar.radio(
    "Jumlah Soal Pilihan Ganda:",
    options=[40, 30],
    index=0
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
delta_thresh = st.sidebar.slider("Sensitivitas Kehitaman Coretan (Delta)", 5, 50, 15, 1)

# ---------------------------------------------------------
# FUNGSI METODE EVALBEE: WARP PERSPECTIVE 4 SUDUT KERTAS
# ---------------------------------------------------------
def order_points(pts):
    """Mengurutkan 4 titik sudut: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def align_and_crop_sheet(image_bytes, target_w=800, target_h=1100):
    """Mengecilkan foto, mendeteksi 4 sudut luar LJK, dan meluruskan kertas miring."""
    raw_pil = Image.open(image_bytes)
    try:
        raw_pil = ImageOps.exif_transpose(raw_pil)
    except Exception:
        pass

    w, h = raw_pil.size
    max_dim = 1000
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        raw_pil = raw_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    img_np = np.array(raw_pil.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is not None:
        pts = screen_cnt.reshape(4, 2)
        rect = order_points(pts)
        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_np, M, (target_w, target_h))
        return warped, True
    else:
        # Fallback jika sudut kertas terpotong frame
        warped = cv2.resize(img_np, (target_w, target_h))
        return warped, False

# ---------------------------------------------------------
# FUNGSI EKSTRAKSI DENSITAS GRID
# ---------------------------------------------------------
def process_evalbee_grid(warped_img, key_answers, total_q=40, sensitivity_delta=15):
    h, w, _ = warped_img.shape
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

    # Koordinat Proposional Area Pilihan Ganda (28% - 58% Tinggi Kertas)
    y1_global = int(h * 0.28)
    y2_global = int(h * 0.58)
    row_h = (y2_global - y1_global) / 10.0

    # Persentase Lebar 4 Kolom Soal (Q1-10, Q11-20, Q21-30, Q31-40)
    col_x_pcts = [
        (0.04, 0.25),
        (0.27, 0.48),
        (0.50, 0.71),
        (0.73, 0.94)
    ]

    col_ranges = [
        range(1, 11),
        range(11, 21),
        range(21, 31),
        range(31, 41)
    ]

    options = ['A', 'B', 'C', 'D']
    detected_answers = {}
    annotated = warped_img.copy()

    for c_idx, q_range in enumerate(col_ranges):
        x_start_pct, x_end_pct = col_x_pcts[c_idx]
        x1_col = int(w * x_start_pct)
        x2_col = int(w * x_end_pct)
        col_w = x2_col - x1_col
        sub_col_w = col_w / 5.0

        # Non-aktifkan kolom 4 jika mode 30 soal
        if q_range[0] > total_q:
            cv2.rectangle(annotated, (x1_col, y1_global), (x2_col, y2_global), (200, 200, 200), -1)
            cv2.putText(annotated, "NON-AKTIF", (x1_col + 10, y1_global + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
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
            "Jawaban Siswa": user_ans,
            "Kunci Jawaban": key_ans,
            "Status": "✅ Benar" if is_correct else ("❌ Salah" if user_ans != "-" else "⚪ Kosong")
        })

    final_score = (score_correct / float(total_q)) * 100.0
    return final_score, score_correct, results, annotated

# ---------------------------------------------------------
# INTERFACE UTAMA: KAMERA NATIVE HP & FILE UPLOAD
# ---------------------------------------------------------
tab_cam, tab_file = st.tabs(["📷 Kamera HP Instan", "📁 Unggah File Gambar"])

captured_file = None

with tab_cam:
    captured_file = st.camera_input("Arahkan kamera ke LJK lalu sentuh tombol Ambil Foto")

with tab_file:
    uploaded_file = st.file_uploader("Pilih gambar dari galeri", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        captured_file = uploaded_file

if captured_file is not None:
    try:
        # 1. Meluruskan Posisi LJK (Warp Perspective Ala EvalBee)
        warped_img, is_warped = align_and_crop_sheet(captured_file, target_w=800, target_h=1100)

        # 2. Evaluasi Densitas Jawaban
        score, correct_count, results, annotated_img = process_evalbee_grid(
            warped_img, key_dict, num_questions, delta_thresh
        )

        if is_warped:
            st.success("⚡ LJK Berhasil Diluruskan & Dipindai Presisi!")
        else:
            st.info("ℹ️ Menggunakan Koreksi Grid Standar LJK SMP YPI.")

        # Tampilan Hasil Visual
        col_v1, col_v2 = st.columns([1, 1])

        with col_v1:
            st.subheader("🔍 Lembar Hasil Scan (Warped)")
            st.image(annotated_img, use_container_width=True)

        with col_v2:
            st.subheader(f"📊 Rekapitulasi Nilai ({num_questions} Soal)")
            st.metric("Nilai Akhir", f"{score:.1f}")
            st.write(f"**Jumlah Benar:** {correct_count} dari {num_questions} Soal")
            st.markdown("---")
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, height=420, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memproses gambar: {str(e)}")
