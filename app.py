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
# GENERATOR SVG OVERLAY: 40 NOMOR SOAL (160 TITIK A-B-C-D)
# ---------------------------------------------------------
def get_svg_overlay_data_uri():
    col_xs = [
        [25, 37, 49, 61],     # Kolom 1: Soal 1-10
        [94, 106, 118, 130],  # Kolom 2: Soal 11-20
        [163, 175, 187, 199], # Kolom 3: Soal 21-30
        [232, 244, 256, 268]  # Kolom 4: Soal 31-40
    ]
    
    # 10 baris vertikal untuk setiap kolom
    row_ys = [118 + i * 27 for i in range(10)]
    
    circles = []
    for col in col_xs:
        for cx in col:
            for cy in row_ys:
                circles.append(f'<circle cx="{cx}" cy="{cy}" r="2.5"/>')
                
    circle_str = "".join(circles)
    
    svg_raw = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 420" width="100%" height="100%">
    <rect x="10" y="8" width="280" height="52" fill="none" stroke="#00FF66" stroke-width="1.5" stroke-dasharray="4,4"/>
    <text x="150" y="38" fill="#00FF66" font-size="11" font-family="sans-serif" text-anchor="middle" font-weight="bold">AREA IDENTITAS / HEADER</text>
    <rect x="10" y="68" width="280" height="342" fill="none" stroke="#00FF66" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="150" y="86" fill="#00FF66" font-size="10" font-family="sans-serif" text-anchor="middle" font-weight="bold">SOAL 1 - 40 (A B C D)</text>
    <g stroke="#00FF66" stroke-width="1" stroke-dasharray="2,2" fill="none" opacity="0.6">
        <rect x="15" y="96" width="62" height="302"/>
        <rect x="84" y="96" width="62" height="302"/>
        <rect x="153" y="96" width="62" height="302"/>
        <rect x="222" y="96" width="62" height="302"/>
    </g>
    <g fill="#00FF66" opacity="0.55">
        {circle_str}
    </g>
    </svg>'''
    
    return urllib.parse.quote(svg_raw)

svg_encoded = get_svg_overlay_data_uri()

# ---------------------------------------------------------
# CSS: FULL SCREEN KAMERA HP & OVERLAY BAYANG-BAYANG
# ---------------------------------------------------------
st.markdown(f"""
    <style>
    .main .block-container {{
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        padding-left: 0.1rem !important;
        padding-right: 0.1rem !important;
        max-width: 100% !important;
    }}

    div[data-testid="stCameraInput"] {{
        position: relative !important;
        width: 100% !important;
        height: 80vh !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        margin: 0 auto;
    }}

    div[data-testid="stCameraInput"] video {{
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        border-radius: 16px !important;
    }}

    div[data-testid="stCameraInput"]::after {{
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 92vw;
        max-width: 460px;
        height: 74vh;
        border: 2px dashed #00FF66;
        border-radius: 12px;
        box-shadow: 0 0 0 2000px rgba(0, 0, 0, 0.65);
        pointer-events: none;
        z-index: 99;
        background-image: url("data:image/svg+xml;utf8,{svg_encoded}");
        background-size: 96% 96%;
        background-position: center;
        background-repeat: no-repeat;
    }}

    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Pemindai LJK SMP YPI")

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
delta_thresh = st.sidebar.slider("Sensitivitas Kehitaman Coretan", 5, 50, 15, 1)

# ---------------------------------------------------------
# METODE EVALBEE: WARP PERSPECTIVE 4 SUDUT KERTAS
# ---------------------------------------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def align_and_crop_sheet(image_bytes, target_w=800, target_h=1100):
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
        warped = cv2.resize(img_np, (target_w, target_h))
        return warped, False

# ---------------------------------------------------------
# FUNGSI EKSTRAKSI GRID PILIHAN GANDA
# ---------------------------------------------------------
def process_evalbee_grid(warped_img, key_answers, total_q=40, sensitivity_delta=15):
    h, w, _ = warped_img.shape
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

    y1_global = int(h * 0.28)
    y2_global = int(h * 0.58)
    row_h = (y2_global - y1_global) / 10.0

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
# INTERFACE UTAMA
# ---------------------------------------------------------
tab_cam, tab_file = st.tabs(["📷 Kamera HP Instan", "📁 Unggah File Gambar"])

captured_file = None

with tab_cam:
    captured_file = st.camera_input("Arahkan LJK sesuai titik-titik hijau di layar")

with tab_file:
    uploaded_file = st.file_uploader("Pilih gambar dari galeri", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        captured_file = uploaded_file

if captured_file is not None:
    try:
        warped_img, is_warped = align_and_crop_sheet(captured_file, target_w=800, target_h=1100)

        score, correct_count, results, annotated_img = process_evalbee_grid(
            warped_img, key_dict, num_questions, delta_thresh
        )

        if is_warped:
            st.success("⚡ LJK Berhasil Diluruskan & Dipindai Presisi!")
        else:
            st.info("ℹ️ Menggunakan Koreksi Grid Standar LJK SMP YPI.")

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
