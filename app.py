import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Scanner LJK Presisi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# FUNGSI 1: AUTO-WARP & PENANGANAN FILE UPLOAD (SCANNED)
# ---------------------------------------------------------
def auto_crop_and_warp(image_np):
    orig = image_np.copy()
    h_orig, w_orig = orig.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edged = cv2.Canny(blur, 75, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Cek apakah menemukan bentuk segi empat (kertas)
        if len(approx) == 4:
            area = cv2.contourArea(c)
            # Jika luas kertas > 20% dari gambar tapi < 95% gambar (berarti ada latar belakang)
            if (h_orig * w_orig * 0.2) < area < (h_orig * w_orig * 0.95):
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                
                dst = np.array([[0, 0], [799, 0], [799, 1099], [0, 1099]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(orig, M, (800, 1100))
                return warped, "Kertas Diluruskan (Auto-Warp)"
                
    # JIKA GAGAL / FILE UPLOAD SUDAH BERUPA SCAN KERTAS PENUH:
    # Jangan potong marginnya, langsung ubah ukuran ke 800x1100 agar presisi
    resized = cv2.resize(orig, (800, 1100))
    return resized, "Resolusi Disesuaikan (Mode Scan Penuh)"

# ---------------------------------------------------------
# FUNGSI 2: PEMROSESAN GRID DENGAN KOORDINAT ABSOLUT
# ---------------------------------------------------------
def process_grid_answers(warped_img, key_answers, config, total_q=40, sensitivity_delta=15):
    h, w, _ = warped_img.shape
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
    annotated = warped_img.copy()
    
    y1_global = int(h * config['y_start'])
    y2_global = int(h * config['y_end'])
    row_h = (y2_global - y1_global) / 10.0

    col_starts = [config['x_col1'], config['x_col2'], config['x_col3'], config['x_col4']]
    col_width = config['col_width']
    
    col_ranges = [range(1, 11), range(11, 21), range(21, 31), range(31, 41)]
    options = ['A', 'B', 'C', 'D']
    detected_answers = {}

    for c_idx, q_range in enumerate(col_ranges):
        x1_col = int(w * col_starts[c_idx])
        x2_col = int(w * (col_starts[c_idx] + col_width))
        sub_col_w = (x2_col - x1_col) / 5.0 

        if q_range[0] > total_q:
            cv2.rectangle(annotated, (x1_col, y1_global), (x2_col, y2_global), (200, 200, 200), -1)
            continue

        for r_idx, q_num in enumerate(q_range):
            if q_num > total_q: break

            row_y1 = int(y1_global + (r_idx * row_h))
            densities = []
            cell_coords = []

            for opt_idx in range(4):
                cx1 = int(x1_col + ((opt_idx + 1) * sub_col_w) + (sub_col_w * 0.15))
                cx2 = int(x1_col + ((opt_idx + 2) * sub_col_w) - (sub_col_w * 0.15))
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

def get_camera_css():
    return """
    <style>
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::before {
        content: ""; position: absolute;
        top: 5%; left: 10%; width: 80%; height: 90%;
        border: 3px solid rgba(0, 255, 0, 0.7);
        z-index: 99; pointer-events: none;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stCameraInput"]::after {
        content: "Paskan Kertas LJK Penuh di Dalam Kotak Ini";
        position: absolute; top: 2%; width: 100%;
        text-align: center; color: #00FF00; font-weight: bold;
        z-index: 99; pointer-events: none;
        text-shadow: 1px 1px 2px #000;
    }
    </style>
    """

# ---------------------------------------------------------
# SETUP STATE UI & PENGATURAN
# ---------------------------------------------------------
st.title("🎯 Pemindai LJK Presisi (Mode Cerdas)")
st.caption("Mendukung Scan Kamera maupun Upload File LJK (PDF/Scan Flatbed).")

if 'num_questions' not in st.session_state: st.session_state['num_questions'] = 40

with st.expander("⚙️ **Kunci Jawaban & Kalibrasi Grid (Geser jika kotak merah meleset)**", expanded=True):
    num_questions = st.radio("Jumlah Soal:", options=[40, 30], index=0 if st.session_state['num_questions'] == 40 else 1, horizontal=True)
    st.session_state['num_questions'] = num_questions

    if 'key_answers_list' not in st.session_state or len(st.session_state['key_answers_list']) != num_questions:
        st.session_state['key_answers_list'] = ['A'] * num_questions

    quick_string = "".join(st.session_state['key_answers_list'])
    user_input = st.text_input(f"Kunci Jawaban ({num_questions} Soal):", value=quick_string).upper()
    cleaned_keys = [char for char in user_input if char in ['A', 'B', 'C', 'D']]
    if len(cleaned_keys) == num_questions:
        st.session_state['key_answers_list'] = cleaned_keys

    st.markdown("---")
    st.markdown("**Kalibrasi Kotak Deteksi** (Gunakan pengaturan ini untuk mempaskan kotak merah/hijau ke huruf ABCD)")
    col_a, col_b = st.columns(2)
    with col_a:
        y_start = st.slider("Batas Atas (Y Start)", 0.15, 0.40, 0.280, 0.005)
        y_end = st.slider("Batas Bawah (Y End)", 0.40, 0.75, 0.580, 0.005)
        col_width = st.slider("Lebar Per Kolom", 0.10, 0.30, 0.210, 0.005)
    with col_b:
        x_col1 = st.slider("Kolom 1 (No 1-10)", 0.00, 0.20, 0.040, 0.005)
        x_col2 = st.slider("Kolom 2 (No 11-20)", 0.20, 0.40, 0.270, 0.005)
        x_col3 = st.slider("Kolom 3 (No 21-30)", 0.40, 0.60, 0.500, 0.005)
        x_col4 = st.slider("Kolom 4 (No 31-40)", 0.60, 0.85, 0.730, 0.005)
    
    delta_thresh = st.slider("Sensitivitas Pensil", 5, 50, 15, 1)

grid_config = {
    'y_start': y_start, 'y_end': y_end, 'col_width': col_width,
    'x_col1': x_col1, 'x_col2': x_col2, 'x_col3': x_col3, 'x_col4': x_col4
}
num_questions = st.session_state['num_questions']
key_dict = {i + 1: st.session_state['key_answers_list'][i] for i in range(num_questions)}

# ---------------------------------------------------------
# INPUT LJK SISWA
# ---------------------------------------------------------
input_method = st.radio("Pilih sumber gambar:", ["Unggah File", "Kamera (Scan Langsung)"], horizontal=True)

siswa_file = None
if input_method == "Unggah File":
    st.info("💡 **Tips Upload:** Pastikan gambar berbentuk tegak lurus. Jika kotak deteksi kurang pas, sesuaikan slider 'Kalibrasi Kotak Deteksi' di menu atas.")
    siswa_file = st.file_uploader("Upload Foto LJK Siswa", type=['jpg', 'jpeg', 'png'])
else:
    st.markdown(get_camera_css(), unsafe_allow_html=True)
    siswa_file = st.camera_input("Ambil Foto")

if siswa_file is not None:
    try:
        raw_siswa = Image.open(siswa_file)
        raw_siswa = ImageOps.exif_transpose(raw_siswa)
        img_siswa_np = np.array(raw_siswa.convert('RGB'))

        with st.spinner("Memproses gambar LJK..."):
            warped_img, status_msg = auto_crop_and_warp(img_siswa_np)
        
        st.success(f"✅ Status Pra-pemrosesan: {status_msg}")

        score, correct_count, results, annotated_img = process_grid_answers(
            warped_img, key_dict, grid_config, num_questions, delta_thresh
        )

        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="📊 NILAI AKHIR", value=f"{score:.1f}")
            st.info(f"**Benar:** {correct_count} dari {num_questions} Soal")
            
            st.subheader("📋 Rincian")
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, height=500, use_container_width=True)
            
        with col_res2:
            st.subheader("🔍 Hasil Deteksi")
            st.caption("Geser perlahan slider kalibrasi di atas jika kotak merah/hijau masih meleset dari abjad.")
            st.image(annotated_img, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
