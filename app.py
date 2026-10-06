import base64
import io
import urllib.parse
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Scanner LJK Presisi - SMP YPI Pulogadung",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# FUNGSI UTILITAS & KONVERSI BASE64
# ---------------------------------------------------------
def pil_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

def adjust_image_with_anchor_offset(img_np, anchor_x, anchor_y, scale, angle, target_w=800, target_h=1100):
    resized_base = cv2.resize(img_np, (target_w, target_h))
    
    # Pusat rotasi & skala disesuaikan dengan titik tengah Anchor Grid (400, 540 pada target resolusi 2x)
    cx, cy = 400, 540
    
    # Membuat matriks transformasi berkebalikan (Inverse) untuk gambar dasar.
    # Jika anchor diputar searah jarum jam (+), foto harus diputar berlawanan (- atau CCW di OpenCV).
    # Jika anchor diperbesar (scale), foto harus diperkecil (1 / scale).
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0 / scale)
    
    # Menerapkan translasi (geser) secara berlawanan
    M[0, 2] -= anchor_x
    M[1, 2] -= anchor_y
    
    shifted = cv2.warpAffine(resized_base, M, (target_w, target_h))
    return shifted

def process_evalbee_grid(warped_img, key_answers, total_q=40, sensitivity_delta=15):
    h, w, _ = warped_img.shape
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

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
# INISIALISASI SESSION STATE
# ---------------------------------------------------------
if 'num_questions' not in st.session_state: st.session_state['num_questions'] = 40
if 'anchor_x' not in st.session_state: st.session_state['anchor_x'] = 0
if 'anchor_y' not in st.session_state: st.session_state['anchor_y'] = 0
if 'anchor_scale' not in st.session_state: st.session_state['anchor_scale'] = 1.0
if 'anchor_angle' not in st.session_state: st.session_state['anchor_angle'] = 0.0

st.title("🎯 Pemindai LJK SMP YPI")

# ---------------------------------------------------------
# ATUR KUNCI JAWABAN
# ---------------------------------------------------------
with st.expander("⚙️ **Atur Kunci Jawaban & Jumlah Soal**", expanded=False):
    num_questions = st.radio("Jumlah Soal:", options=[40, 30], index=0 if st.session_state['num_questions'] == 40 else 1, horizontal=True)
    st.session_state['num_questions'] = num_questions

    if 'key_answers_list' not in st.session_state or len(st.session_state['key_answers_list']) != num_questions:
        st.session_state['key_answers_list'] = ['A'] * num_questions

    tab_edit1, tab_edit2 = st.tabs(["⚡ Input Cepat", "📊 Tabel Edit"])
    with tab_edit1:
        quick_string = "".join(st.session_state['key_answers_list'])
        user_input = st.text_input(f"Ketik {num_questions} Kunci (Contoh: ABCD...):", value=quick_string).upper()
        cleaned_keys = [char for char in user_input if char in ['A', 'B', 'C', 'D']]
        if len(cleaned_keys) == num_questions:
            st.session_state['key_answers_list'] = cleaned_keys

    with tab_edit2:
        df_keys = pd.DataFrame({"No": list(range(1, num_questions + 1)), "Kunci": st.session_state['key_answers_list']})
        edited_df = st.data_editor(
            df_keys,
            column_config={"No": st.column_config.NumberColumn("No", disabled=True), "Kunci": st.column_config.SelectboxColumn("Kunci", options=['A', 'B', 'C', 'D'], required=True)},
            hide_index=True, use_container_width=True, height=220
        )
        st.session_state['key_answers_list'] = edited_df["Kunci"].tolist()

    delta_thresh = st.slider("Sensitivitas Kehitaman Pensil", 5, 50, 15, 1)

num_questions = st.session_state['num_questions']
key_dict = {i + 1: st.session_state['key_answers_list'][i] for i in range(num_questions)}

# ---------------------------------------------------------
# UPLOAD GAMBAR & CANVAS GESER ANCHOR PG
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload foto LJK dari Galeri", type=['jpg', 'jpeg', 'png'])
processed_img = None

if uploaded_file is not None:
    raw_pil = Image.open(uploaded_file)
    try: raw_pil = ImageOps.exif_transpose(raw_pil)
    except Exception: pass

    img_base64 = pil_to_base64(raw_pil)

    st.markdown("### 🎯 Sesuaikan Grid LJK dengan Interaktif")
    st.caption("Pilih mode di bawah, lalu **Klik & Tahan (atau usap touchpad)** pada foto untuk memanipulasi kisi hijau.")

    # CANVAS INTERAKTIF: MOVE, SCALE, ROTATE DENGAN DRAG MOUSE/TOUCHPAD
    html_canvas_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; font-family: sans-serif;">
        <!-- Panel Mode Kontrol -->
        <div style="background: #2a2a2a; padding: 10px 20px; border-radius: 8px; margin-bottom: 15px; color: white; display: flex; gap: 20px;">
            <label style="cursor: pointer;"><input type="radio" name="mode" value="move" checked> ✋ Geser X/Y</label>
            <label style="cursor: pointer;"><input type="radio" name="mode" value="scale"> 🔍 Skala (Naik/Turun)</label>
            <label style="cursor: pointer;"><input type="radio" name="mode" value="rotate"> 🔄 Putar (Kiri/Kanan)</label>
        </div>

        <canvas id="dragAnchorCanvas" width="400" height="550" style="border: 2px solid #00FF66; border-radius: 12px; cursor: move; touch-action: none; background: #111;"></canvas>
        
        <div style="margin-top: 15px; color: #333; font-size: 14px; background: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; width: 100%; max-width: 400px;">
            <b>X:</b> <span id="lblX">0</span> | <b>Y:</b> <span id="lblY">0</span> | <b>Skala:</b> <span id="lblS">1.00</span> | <b>Rotasi:</b> <span id="lblR">0.0</span>°
        </div>
    </div>

    <script>
        const canvas = document.getElementById('dragAnchorCanvas');
        const ctx = canvas.getContext('2d');
        const lblX = document.getElementById('lblX');
        const lblY = document.getElementById('lblY');
        const lblS = document.getElementById('lblS');
        const lblR = document.getElementById('lblR');
        const modeRadios = document.querySelectorAll('input[name="mode"]');

        let img = new Image();
        img.src = "{img_base64}";

        let isDragging = false;
        let startX = 0, startY = 0;
        
        // State Awal dari Streamlit
        let anchorX = {st.session_state['anchor_x']};
        let anchorY = {st.session_state['anchor_y']};
        let anchorScale = {st.session_state['anchor_scale']};
        let anchorAngle = {st.session_state['anchor_angle']};

        img.onload = function() {{ draw(); }};

        function getMode() {{
            for (const radio of modeRadios) {{
                if (radio.checked) return radio.value;
            }}
            return 'move';
        }}

        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            ctx.save();
            
            // 1. Translasi Posisi Anchor
            ctx.translate(anchorX, anchorY);

            // 2. Pusat Rotasi & Skala (Kira-kira di tengah Grid Hijau)
            let centerX = 200;
            let centerY = 270;
            
            ctx.translate(centerX, centerY);
            ctx.rotate(anchorAngle * Math.PI / 180);
            ctx.scale(anchorScale, anchorScale);
            ctx.translate(-centerX, -centerY);

            // Gambar Kisi-Kisi Anchor
            ctx.strokeStyle = "#00FF66";
            ctx.lineWidth = 2;
            
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(15, 10, 370, 70); // Header
            
            ctx.setLineDash([3, 3]);
            ctx.strokeRect(15, 90, 370, 440); // Body Soal

            ctx.setLineDash([]);
            ctx.fillStyle = "#00FF66";
            for (let col = 0; col < 4; col++) {{
                let colX = 35 + col * 90;
                for (let row = 0; row < 10; row++) {{
                    let rowY = 125 + row * 38;
                    for (let opt = 0; opt < 4; opt++) {{
                        ctx.beginPath();
                        ctx.arc(colX + opt * 18, rowY, 3.5, 0, 2 * Math.PI);
                        ctx.fill();
                    }}
                }}
            }}
            ctx.restore();

            lblX.innerText = Math.round(anchorX);
            lblY.innerText = Math.round(anchorY);
            lblS.innerText = anchorScale.toFixed(2);
            lblR.innerText = anchorAngle.toFixed(1);
        }}

        // EVENT DRAG (Mouse / Touchpad)
        function handleDragStart(x, y) {{
            isDragging = true;
            startX = x;
            startY = y;
            canvas.style.cursor = 'grabbing';
        }}

        function handleDragMove(x, y) {{
            if (!isDragging) return;
            let dx = x - startX;
            let dy = y - startY;
            let mode = getMode();

            if (mode === 'move') {{
                anchorX += dx;
                anchorY += dy;
            }} else if (mode === 'scale') {{
                anchorScale -= dy * 0.005; // Tarik ke atas membesar, bawah mengecil
                if(anchorScale < 0.5) anchorScale = 0.5;
                if(anchorScale > 2.0) anchorScale = 2.0;
            }} else if (mode === 'rotate') {{
                anchorAngle += dx * 0.3; // Tarik kanan putar kanan, kiri putar kiri
            }}

            startX = x;
            startY = y;
            draw();
        }}

        function handleDragEnd() {{
            isDragging = false;
            canvas.style.cursor = 'move';
        }}

        canvas.addEventListener('mousedown', (e) => handleDragStart(e.clientX, e.clientY));
        window.addEventListener('mousemove', (e) => handleDragMove(e.clientX, e.clientY));
        window.addEventListener('mouseup', handleDragEnd);

        canvas.addEventListener('touchstart', (e) => handleDragStart(e.touches[0].clientX, e.touches[0].clientY));
        canvas.addEventListener('touchmove', (e) => handleDragMove(e.touches[0].clientX, e.touches[0].clientY));
        canvas.addEventListener('touchend', handleDragEnd);
    </script>
    """

    components.html(html_canvas_code, height=680)

    st.warning("⚠️ **PENTING:** Salin angka dari indikator abu-abu di atas ke panel di bawah ini untuk memproses gambar!")
    
    # Sinkronisasi Koordinat Anchor dari HTML ke Python Backend
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.session_state['anchor_x'] = st.number_input("Posisi X", value=st.session_state['anchor_x'], step=2)
    with col2: st.session_state['anchor_y'] = st.number_input("Posisi Y", value=st.session_state['anchor_y'], step=2)
    with col3: st.session_state['anchor_scale'] = st.number_input("Skala", value=st.session_state['anchor_scale'], step=0.01, format="%.2f")
    with col4: st.session_state['anchor_angle'] = st.number_input("Rotasi (°)", value=st.session_state['anchor_angle'], step=0.5, format="%.1f")

    img_np = np.array(raw_pil.convert('RGB'))
    
    # Pemotongan akurat berdasarkan input tersinkronisasi
    processed_img = adjust_image_with_anchor_offset(
        img_np, 
        st.session_state['anchor_x'] * 2, 
        st.session_state['anchor_y'] * 2,
        st.session_state['anchor_scale'],
        st.session_state['anchor_angle']
    )

# ---------------------------------------------------------
# HASIL ANALISIS
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
