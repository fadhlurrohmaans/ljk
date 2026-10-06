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
# GENERATOR SVG OVERLAY
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

if 'num_questions' not in st.session_state:
    st.session_state['num_questions'] = 40

st.title("🎯 Pemindai LJK SMP YPI")

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
# FUNGSI PERGESERAN & PENYESUAIAN POSISI
# ---------------------------------------------------------
def adjust_image_position(img_np, shift_x, shift_y, zoom_scale, target_w=800, target_h=1100):
    h, w, _ = img_np.shape
    resized_base = cv2.resize(img_np, (target_w, target_h))
    
    # Terapkan Zoom / Skala
    if zoom_scale != 1.0:
        new_w = int(target_w * zoom_scale)
        new_h = int(target_h * zoom_scale)
        scaled_img = cv2.resize(resized_base, (new_w, new_h))
        
        # Crop / Pad kembali ke target resolution
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        start_x_src = max(0, (new_w - target_w) // 2)
        start_y_src = max(0, (new_h - target_h) // 2)
        end_x_src = min(new_w, start_x_src + target_w)
        end_y_src = min(new_h, start_y_src + target_h)
        
        start_x_dst = max(0, (target_w - new_w) // 2)
        start_y_dst = max(0, (target_h - new_h) // 2)
        end_x_dst = min(target_w, start_x_dst + (end_x_src - start_x_src))
        end_y_dst = min(target_h, start_y_dst + (end_y_src - start_y_src))
        
        canvas[start_y_dst:end_y_dst, start_x_dst:end_x_dst] = scaled_img[start_y_src:end_y_src, start_x_src:end_x_src]
        resized_base = canvas

    # Terapkan Pergeseran Posisi (Translation X & Y)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(resized_base, M, (target_w, target_h))
    return shifted

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
# TAB AMBIL GAMBAR & PENYESUAIAN SLIDER
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
        
        img_np = np.array(raw_pil.convert('RGB'))

        st.subheader("🎛️ Pasang & Geser Posisi Anchor LJK")
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        with col_ctrl1:
            shift_x = st.slider("↔️ Geser Kiri / Kanan", -150, 150, 0, 2)
        with col_ctrl2:
            shift_y = st.slider("↕️ Geser Atas / Bawah", -150, 150, 0, 2)
        with col_ctrl3:
            zoom = st.slider("🔍 Perbesar / Perkecil", 0.7, 1.3, 1.0, 0.02)

        processed_img = adjust_image_position(img_np, shift_x, shift_y, zoom)

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
