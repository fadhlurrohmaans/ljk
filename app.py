import io
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(page_title="Scanner LJK 1-Foto Presisi - SMP YPI Pulogadung", layout="wide")

st.title("🎯 Pemindai LJK Presisi ( cukup 1x Foto Full LJK )")
st.caption("Metode Perspective Warping & Fixed Grid Projection - SMP YPI Pulogadung (40 Soal)")

# ---------------------------------------------------------
# FUNGSI KOMPRESI & OTOMATISASI ORIENTASI
# ---------------------------------------------------------
def optimize_input_image(file_bytes, max_dim=1200):
    """Mengecilkan foto HP dan merapikan orientasi EXIF."""
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

# ---------------------------------------------------------
# PERSPECTIVE WARPING: MELURUSKAN FOTO LJK KEMBALI TEGAK
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

def warp_ljk_sheet(image_np, target_w=800, target_h=1100):
    """Mendeteksi batas luar LJK dan melakukan transformasi ortogonal ke kanvas 800x1100."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
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

    # Jika kontur 4 sudut terdeteksi, lakukan warping perspektif
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
        warped = cv2.warpPerspective(image_np, M, (target_w, target_h))
        return warped, True
    else:
        # Fallback: Resize langsung ke kanvas target jika LJK memenuhi frame
        warped = cv2.resize(image_np, (target_w, target_h))
        return warped, False

# ---------------------------------------------------------
# FIXED GRID PROJECTION: EVALUASI JAWABAN BERDASARKAN KOORDINAT TERTENTU
# ---------------------------------------------------------
def evaluate_fixed_grid(warped_img, key_answers, sensitivity_delta=15):
    h, w, _ = warped_img.shape
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

    # Area Y untuk 10 Baris Pilihan Ganda (Persentase terhadap tinggi kertas)
    y_start_pct, y_end_pct = 0.28, 0.58
    y1_global = int(h * y_start_pct)
    y2_global = int(h * y_end_pct)
    roi_h = y2_global - y1_global
    row_h = roi_h / 10.0

    # Persentase X untuk 4 Kolom Soal (Q1-10, Q11-20, Q21-30, Q31-40)
    col_x_pcts = [
        (0.05, 0.25),  # Kolom 1
        (0.28, 0.48),  # Kolom 2
        (0.51, 0.71),  # Kolom 3
        (0.74, 0.94)   # Kolom 4
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
        sub_col_w = col_w / 5.0  # Sub-kolom 0: No, 1-4: A, B, C, D

        for r_idx, q_num in enumerate(q_range):
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

            # Tentukan warna bounding box visualisasi
            for opt_idx, (cx1, cy1, cx2, cy2) in enumerate(cell_coords):
                if opt_idx == max_idx and selected_option != "-":
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (200, 200, 200), 1)

    return detected_answers, annotated

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
st.sidebar.header("🎛️ Sensitivitas Deteksi")
delta_thresh = st.sidebar.slider("Kontras Kehitaman Coretan (Delta)", 5, 50, 15, 1)

# Pengunggah Single Foto
uploaded_file = st.file_uploader(
    "📷 Ambil 1 Foto Utuh LJK via Kamera HP / Unggah Gambar", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    try:
        # 1. Kompresi & penyesuaian rotasi di RAM
        raw_img = optimize_input_image(uploaded_file, max_dim=1200)

        # 2. Pelurusan Kertas LJK (Warp Perspective)
        warped_img, is_warped = warp_ljk_sheet(raw_img, target_w=800, target_h=1100)

        # 3. Evaluasi Jawaban pada Kanvas Terstandarisasi
        detected_answers, annotated_img = evaluate_fixed_grid(warped_img, key_dict, delta_thresh)

        if is_warped:
            st.success("✅ Sudut kertas LJK berhasil terdeteksi, diluruskan, dan dipindai secara presisi!")
        else:
            st.info("ℹ️ Kertas diproses menggunakan pencocokan grid standar.")

        # Display Visual Hasil Scan
        st.subheader("🔍 Visualisasi Pemindaian Utuh")
        st.image(annotated_img, caption="Hasil Deteksi Jawaban (Kotak Hijau)", use_container_width=True)

        # Perhitungan Nilai Akhir
        score_correct = 0
        results = []
        for q_num in range(1, 41):
            user_ans = detected_answers.get(q_num, "-")
            key_ans = key_dict.get(q_num, "A")

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

        st.markdown("---")
        st.subheader("📊 Rekapitulasi Nilai")
        c_m1, c_m2 = st.columns([1, 2])
        with c_m1:
            st.metric("Nilai Akhir", f"{final_score:.1f}")
            st.write(f"**Jumlah Benar:** {score_correct} dari 40 Soal")
        with c_m2:
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, height=300, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memproses gambar: {str(e)}")
