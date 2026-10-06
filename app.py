import io
import base64
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Scanner LJK Presisi (Anchor Dual-Point)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# FUNGSI 1: AUTO-WARP & PERSPECTIVE TRANSFORM
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
        
        if len(approx) == 4:
            area = cv2.contourArea(c)
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
                
    resized = cv2.resize(orig, (800, 1100))
    return resized, "Resolusi Disesuaikan (Mode Scan Penuh)"

# ---------------------------------------------------------
# FUNGSI 2: PEMROSESAN GRID DENGAN ANCHOR NO.1-A & NO.40-D
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

            # Visualisasi Jawaban Siswa
            for opt_idx, (cx1, cy1, cx2, cy2) in enumerate(cell_coords):
                if opt_idx == max_idx and selected_option != "-":
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)

                # --- ANCHOR KHUSUS NO. 1 OPTION A ---
                if q_num == 1 and opt_idx == 0:
                    center_x, center_y = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                    cv2.circle(annotated, (center_x, center_y), 14, (0, 255, 255), 2) # Lingkaran Kuning
                    cv2.drawMarker(annotated, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(annotated, "ANCHOR 1-A", (cx1 - 25, cy1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

                # --- ANCHOR KHUSUS NO. 40 OPTION D ---
                if q_num == 40 and opt_idx == 3:
                    center_x, center_y = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                    cv2.circle(annotated, (center_x, center_y), 14, (0, 255, 255), 2) # Lingkaran Kuning
                    cv2.drawMarker(annotated, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(annotated, "ANCHOR 40-D", (cx1 - 30, cy2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

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
# FUNGSI 3: GENERATOR PANDUAN AR (SVG) KAMERA DENGAN ANCHOR
# ---------------------------------------------------------
def get_camera_guide_css(config, total_q=40):
    w, h_img = 800, 1100
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h_img}">'
    
    # Anchor Sudut Kertas Outer Frame
    svg += '<rect x="15" y="15" width="50" height="50" fill="none" stroke="#00FF00" stroke-width="3"/>'
    svg += '<rect x="735" y="15" width="50" height="50" fill="none" stroke="#00FF00" stroke-width="3"/>'
    svg += '<rect x="15" y="1035" width="50" height="50" fill="none" stroke="#00FF00" stroke-width="3"/>'
    svg += '<rect x="735" y="1035" width="50" height="50" fill="none" stroke="#00FF00" stroke-width="3"/>'
    
    y1 = int(h_img * config['y_start'])
    y2 = int(h_img * config['y_end'])
    row_h = (y2 - y1) / 10.0
    col_starts = [config['x_col1'], config['x_col2'], config['x_col3'], config['x_col4']]
    col_width = config['col_width']

    for c_idx, start_pct in enumerate(col_starts):
        if (c_idx * 10) >= total_q: break
        
        x1 = w * start_pct
        x2 = w * (start_pct + col_width)
        col_w = x2 - x1
        sub_col_w = col_w / 5.0
        
        # Garis Kolom
        svg += f'<rect x="{x1}" y="{y1}" width="{col_w}" height="{row_h*10}" fill="none" stroke="#00FF00" stroke-width="1.5" stroke-dasharray="4,4" opacity="0.5"/>'
        
        for r in range(10):
            q_num = c_idx * 10 + r + 1
            if q_num > total_q: break
            row_y = y1 + (r * row_h)
            
            for opt in range(4):
                cx = x1 + ((opt + 1.5) * sub_col_w)
                cy = row_y + (row_h / 2)
                
                # OPTION 1-A (ANCHOR ATAS-KIRI)
                if q_num == 1 and opt == 0:
                    svg += f'<circle cx="{cx}" cy="{cy}" r="14" fill="rgba(255,255,0,0.4)" stroke="#FFFF00" stroke-width="3"/>'
                    svg += f'<line x1="{cx-20}" y1="{cy}" x2="{cx+20}" y2="{cy}" stroke="#FFFF00" stroke-width="2"/>'
                    svg += f'<line x1="{cx}" y1="{cy-20}" x2="{cx}" y2="{cy+20}" stroke="#FFFF00" stroke-width="2"/>'
                    svg += f'<text x="{cx-35}" y="{cy-22}" fill="#FFFF00" font-size="16" font-weight="bold">🎯 ANCHOR 1-A</text>'
                
                # OPTION 40-D (ANCHOR BAWAH-KANAN)
                elif q_num == 40 and opt == 3:
                    svg += f'<circle cx="{cx}" cy="{cy}" r="14" fill="rgba(255,255,0,0.4)" stroke="#FFFF00" stroke-width="3"/>'
                    svg += f'<line x1="{cx-20}" y1="{cy}" x2="{cx+20}" y2="{cy}" stroke="#FFFF00" stroke-width="2"/>'
                    svg += f'<line x1="{cx}" y1="{cy-20}" x2="{cx}" y2="{cy+20}" stroke="#FFFF00" stroke-width="2"/>'
                    svg += f'<text x="{cx-45}" y="{cy+35}" fill="#FFFF00" font-size="16" font-weight="bold">🎯 ANCHOR 40-D</text>'
                
                else:
                    svg += f'<circle cx="{cx}" cy="{cy}" r="5" fill="rgba(0,255,0,0.3)" stroke="#00FF00" stroke-width="1.5"/>'
                
    svg += '</svg>'
    
    b64_svg = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    
    css = f"""
    <style>
    [data-testid="stCameraInput"] {{
        position: relative;
    }}
    [data-testid="stCameraInput"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        margin: auto; width: 100%; height: 100%;
        background-image: url('data:image/svg+xml;base64,{b64_svg}');
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        z-index: 99;
        pointer-events: none;
        background-color: rgba(0,0,0,0.15);
    }}
    [data-testid="stCameraInput"]::after {{
        content: "Paskan ANCHOR 1-A (Atas Kiri) & ANCHOR 40-D (Bawah Kanan)";
        position: absolute; top: 10px; left: 0; width: 100%;
        text-align: center; color: #FFFF00;
        font-weight: bold; font-size: 15px;
        z-index: 99; pointer-events: none;
        text-shadow: 2px 2px 4px #000;
    }}
    </style>
    """
    return css

# ---------------------------------------------------------
# SETUP STATE UI & PENGATURAN
# ---------------------------------------------------------
st.title("🎯 Pemindai LJK Presisi (Dual-Anchor System)")
st.caption("Memakai patokan khusus pada No. 1 (A) & No. 40 (D) untuk akurasi pemindaian maksimal.")

if 'num_questions' not in st.session_state: st.session_state['num_questions'] = 40

with st.expander("⚙️ **Kunci Jawaban & Kalibrasi Grid**", expanded=False):
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
    st.markdown("**Kalibrasi Posisi Grid**")
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
    st.info("💡 **Petunjuk:** Perhatikan tanda **ANCHOR 1-A** dan **ANCHOR 40-D** berwarna kuning pada gambar hasil deteksi. Pastikan keduanya pas menimpa bulatan LJK.")
    siswa_file = st.file_uploader("Upload Foto LJK Siswa", type=['jpg', 'jpeg', 'png'])
else:
    # Tampilkan overlay kamera AR dengan Anchor 1-A dan 40-D
    st.markdown(get_camera_guide_css(grid_config, num_questions), unsafe_allow_html=True)
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
            
            st.subheader("📋 Rincian Jawaban")
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, height=500, use_container_width=True)
            
        with col_res2:
            st.subheader("🔍 Hasil Deteksi & Anchor Verification")
            st.caption("Akurasi terjamin jika target kuning 'ANCHOR 1-A' (kiri atas) dan 'ANCHOR 40-D' (kanan bawah) tepat berada di atas bulatan LJK.")
            st.image(annotated_img, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
