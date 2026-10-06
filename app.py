import io
import urllib.parse
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper

st.set_page_config(
    page_title="Scanner LJK Presisi - SMP YPI Pulogadung",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# GENERATOR SVG OVERLAY (LAYAR KAMERA)
# ---------------------------------------------------------
def get_svg_overlay_data_uri():
    col_xs = [
        [25, 37, 49, 61],
        [94, 106, 118, 130],
        [163, 175, 187, 199],
        [232, 244, 256, 268]
    ]
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
# INISIALISASI SESSION STATE
# ---------------------------------------------------------
if 'num_questions' not in st.session_state:
    st.session_state['num_questions'] = 40

st.title("🎯 Pemindai LJK SMP YPI")

is_fullscreen = st.toggle("📱 Mode Kamera Full Screen (Layar Penuh)", value=False)

cam_height = "85vh" if is_fullscreen else "60vh"
cam_max_h = "none" if is_fullscreen else "520px"
overlay_h = "78vh" if is_fullscreen else "52vh"
overlay_max_h = "none" if is_fullscreen else "440px"

# ---------------------------------------------------------
# CSS RESPONSIF
# ---------------------------------------------------------
st.markdown(f"""
    <style>
    .main .block-container {{
        padding-top: 0.2rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.3rem !important;
        padding-right: 0.3rem !important;
        max-width: 100% !important;
    }}

    h1 {{
        font-size: 1.4rem !important;
        text-align: center;
        margin-bottom: 0.2rem !important;
    }}

    div[data-testid="stCameraInput"] {{
        position: relative !important;
        width: 100% !important;
        max-width: 500px !important;
        height: {cam_height} !important;
        min-height: 380px !important;
        max-height: {cam_max_h} !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        margin: 0 auto !important;
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
        max-width: 420px;
        height: {overlay_h};
        max-height: {overlay_max_h};
        border: 2px dashed #00FF66;
        border-radius: 12px;
        box-shadow: 0 0 0 2000px rgba(0, 0, 0, 0.65);
        pointer-events: none;
        z-index: 10;
        background-image: url("data:image/svg+xml;utf8,{svg_encoded}");
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .stButton button, .stDownloadButton button {{
        width: 100% !important;
        min-height: 46px !important;
        font-size: 16px !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }}

    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ATUR KUNCI JAWABAN
# ---------------------------------------------------------
with st.expander("⚙️ **Atur Kunci Jawaban & Jumlah Soal**", expanded=False):
    num_questions = st.radio(
        "Jumlah Soal:",
        options=[40, 30],
        index=0 if st.session_state['num_questions'] == 40 else 1,
        horizontal=True
    )
    st.session_state['num_questions'] = num_questions

    if 'key_answers_list' not in st.session_state or len(st.session_state['key_answers_list']) != num_questions:
        st.session_state['key_answers_list'] = ['A'] * num_questions

    tab_edit1, tab_edit2 = st.tabs(["⚡ Input Cepat", "📊 Tabel Edit"])

    with tab_edit1:
        quick_string = "".join(st.session_state['key_answers_list'])
        user_input = st.text_input(
            f"Ketik {num_questions} Kunci (Contoh: ABCD...):",
            value=quick_string
        ).upper()
        
        cleaned_keys = [char for char in user_input if char in ['A', 'B', 'C', 'D']]
        if len(cleaned_keys) == num_questions:
            st.session_state['key_answers_list'] = cleaned_keys
            st.success(f"✅ Kunci {num_questions} soal tersimpan!")
        elif len(user_input) > 0:
            st.warning(f"Terdeteksi {len(cleaned_keys)}/{num_questions} kunci valid (A/B/C/D).")

    with tab_edit2:
        df_keys = pd.DataFrame({
            "No": list(range(1, num_questions + 1)),
            "Kunci": st.session_state['key_answers_list']
        })
        
        edited_df = st.data_editor(
            df_keys,
            column_config={
                "No": st.column_config.NumberColumn("No", disabled=True),
                "Kunci": st.column_config.SelectboxColumn("Kunci", options=['A', 'B', 'C', 'D'], required=True)
            },
            hide_index=True,
            use_container_width=True,
            height=220
        )
        st.session_state['key_answers_list'] = edited_df["Kunci"].tolist()

    delta_thresh = st.slider("Sensitivitas Kehitaman Pensil", 5, 50, 15, 1)

num_questions = st.session_state['num_questions']
key_dict = {i + 1: st.session_state['key_answers_list'][i] for i in range(num_questions)}

# ---------------------------------------------------------
# FUNGSI PROSES GRID EVALBEE
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
# TAB AMBIL GAMBAR / UPLOAD GALERI
# ---------------------------------------------------------
tab_cam, tab_file = st.tabs(["📷 Ambil Foto LJK", "📁 Upload Galeri"])

processed_img = None

with tab_cam:
    captured_file = st.camera_input("Posisikan LJK di dalam garis hijau")
    if captured_file is not None:
        raw_pil = Image.open(captured_file)
        try:
            raw_pil = ImageOps.exif_transpose(raw_pil)
        except Exception:
            pass
        img_np = np.array(raw_pil.convert('RGB'))
        processed_img = cv2.resize(img_np, (800, 1100))

with tab_file:
    uploaded_file = st.file_uploader("Pilih foto LJK dari Galeri HP", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        raw_pil = Image.open(uploaded_file)
        try:
            raw_pil = ImageOps.exif_transpose(raw_pil)
        except Exception:
            pass
        
        st.info("👆 **Geser kotak hijau di bawah ini** hingga menutupi seluruh area lembar LJK, lalu sistem akan otomatis menyesuaikan posisi anchor.")
        
        # INTERAKSI GESER LANGSUNG PADA GAMBAR (STREAMLIT CROPPER)
        cropped_pil = st_cropper(
            raw_pil,
            realtime_update=True,
            box_color='#00FF66',
            aspect_ratio=(8, 11)
        )
        
        img_np = np.array(cropped_pil.convert('RGB'))
        processed_img = cv2.resize(img_np, (800, 1100))

# ---------------------------------------------------------
# TAMPILAN HASIL SCAN
# ---------------------------------------------------------
if processed_img is not None:
    try:
        score, correct_count, results, annotated_img = process_evalbee_grid(
            processed_img, key_dict, num_questions, delta_thresh
        )

        st.markdown("---")
        
        st.metric(label="📊 NILAI AKHIR", value=f"{score:.1f}")
        st.info(f"**Jawaban Benar:** {correct_count} dari {num_questions} Soal")

        st.subheader("🔍 Hasil Analisis LJK")
        st.image(annotated_img, use_container_width=True)

        st.subheader("📋 Rincian Jawaban Per Nomor")
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, height=350, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memproses LJK: {str(e)}")
