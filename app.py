import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Auto-Scan LJK EvalBee - SMP YPI Pulogadung", layout="wide")

# ---------------------------------------------------------
# CSS: BINGKAI PANDUAN KAMERA HP UKURAN FULL / LEBIH LEUASA
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Pembungkus Kamera */
    div[data-testid="stCameraInput"] {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Overlay Bingkai Target Scanner Ukuran Luas (Full Grid) */
    div[data-testid="stCameraInput"]::before {
        content: "🎯 PASIKAN KERTAS / TABEL LJK DI DALAM BINGKAI HIJAU";
        position: absolute;
        top: 6%;
        left: 4%;
        width: 92%;
        height: 84%;
        border: 3px dashed #00FF00;
        border-radius: 12px;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.40);
        z-index: 99;
        pointer-events: none;
        color: #00FF00;
        font-weight: bold;
        font-size: 14px;
        text-align: center;
        padding-top: 10px;
        background: rgba(0, 255, 0, 0.05);
        box-sizing: border-box;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📱 Pemindai LJK Otomatis (EvalBee Mode)")
st.caption("Khusus Format LJK SMP YPI Pulogadung (40 Soal Pilihan Ganda)")

# ---------------------------------------------------------
# FUNGSI PERSPECTIVE TRANSFORM (MELURUSKAN GAMBAR OTOMATIS)
# ---------------------------------------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left
    return rect

def auto_align_table(image_np):
    """Mencari kontur tabel Pilihan Ganda & meratakannya secara otomatis."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    table_corner = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Deteksi kontur segi empat dengan toleransi area lebih fleksibel
        if len(approx) == 4 and cv2.contourArea(c) > (image_np.shape[0] * image_np.shape[1] * 0.05):
            table_corner = approx.reshape(4, 2)
            break
            
    if table_corner is not None:
        pts1 = order_points(table_corner)
        maxWidth, maxHeight = 1200, 520
        pts2 = np.float32([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(image_np, M, (maxWidth, maxHeight))
        return warped, True
    else:
        # Fallback Crop Luas (18% - 58% tinggi gambar & 2% - 98% lebar gambar)
        h, w, _ = image_np.shape
        crop = image_np[int(h*0.18):int(h*0.58), int(w*0.02):int(w*0.98)]
        resized = cv2.resize(crop, (1200, 520))
        return resized, False

# ---------------------------------------------------------
# FUNGSI EVALUASI GRID JAWABAN (4 KOLOM x 10 BARIS x 5 SEL)
# ---------------------------------------------------------
def process_warped_ljk(warped_img, key_answers, sensitivity=110, min_pixels=45):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY_INV)
    
    h, w = binary.shape
    col_w = w / 4.0
    row_h = h / 10.0
    
    col_ranges = [
        range(1, 11),
        range(11, 21),
        range(21, 31),
        range(31, 41)
    ]
    options = ['A', 'B', 'C', 'D']
    
    detected_answers = {}
    annotated_img = warped_img.copy()
    
    for c_idx, q_range in enumerate(col_ranges):
        col_x_start = c_idx * col_w
        
        for r_idx, q_num in enumerate(q_range):
            row_y_start = r_idx * row_h
            sub_col_w = col_w / 5.0
            
            max_pixels = 0
            selected_option = "-"
            
            for opt_idx, opt_label in enumerate(options):
                x1 = int(col_x_start + ((opt_idx + 1) * sub_col_w) + (sub_col_w * 0.18))
                x2 = int(col_x_start + ((opt_idx + 2) * sub_col_w) - (sub_col_w * 0.18))
                y1 = int(row_y_start + (row_h * 0.18))
                y2 = int(row_y_start + row_h - (row_h * 0.18))
                
                cell = binary[y1:y2, x1:x2]
                pixel_count = cv2.countNonZero(cell) if cell.size > 0 else 0
                
                color = (0, 255, 0) if pixel_count > min_pixels else (200, 200, 200)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 1)
                
                if pixel_count > min_pixels and pixel_count > max_pixels:
                    max_pixels = pixel_count
                    selected_option = opt_label
                    
            detected_answers[q_num] = selected_option

    score_correct = 0
    results = []
    
    for q_num in range(1, 41):
        user_ans = detected_answers.get(q_num, "-")
        key_ans = key_answers.get(q_num, "A")
        
        is_correct = (user_ans == key_ans)
        if is_correct:
            score_correct += 1
            
        results.append({
            "No": q_num,
            "Jawaban": user_ans,
            "Kunci": key_ans,
            "Hasil": "✅" if is_correct else "❌"
        })
        
    final_score = (score_correct / 40.0) * 100.0
    return final_score, score_correct, results, annotated_img

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
st.sidebar.header("🎛️ Ambang Toleransi")
threshold_val = st.sidebar.slider("Kehitaman Pensil/Coretan", 50, 200, 110)
min_pixel_val = st.sidebar.slider("Ukuran Coretan Minimal", 20, 150, 45)

option_input = st.radio("Pilih Metode Pemindaian:", ["📷 Pakai Kamera HP Live", "📁 Unggah File Foto"], horizontal=True)

uploaded_file = None

if option_input == "📷 Pakai Kamera HP Live":
    uploaded_file = st.camera_input("Arahkan kamera ke kertas LJK")
else:
    uploaded_file = st.file_uploader("Unggah file foto LJK (JPG / PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file)
        img_np = np.array(pil_image.convert('RGB'))
        
        warped_grid, is_auto = auto_align_table(img_np)
        
        score, correct_count, results, annotated_grid = process_warped_ljk(
            warped_grid, key_dict, threshold_val, min_pixel_val
        )
        
        if is_auto:
            st.success("✅ Tabel LJK berhasil dideteksi dan diluruskan secara otomatis!")
        else:
            st.info("ℹ️ Menggunakan pemotongan area luas (Full-Range Fallback Mode).")

        c1, c2 = st.columns([1.2, 1])
        
        with c1:
            st.subheader("🔍 Hasil Auto-Crop & Alignment")
            st.image(annotated_grid, use_container_width=True, caption="Kotak Hijau = Coretan 'X' Terbaca Sistem")
            
        with c2:
            st.subheader("📊 Ringkasan Nilai")
            st.metric("Nilai Akhir", f"{score:.1f}")
            st.write(f"**Jumlah Benar:** {correct_count} / 40 Soal")
            
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, height=350, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memproses gambar: {str(e)}")
