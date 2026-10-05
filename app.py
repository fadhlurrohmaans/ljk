import traceback
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(page_title="Koreksi LJK Otomatis", layout="wide")

try:
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageOps

    st.title("📋 Pemindai & Koreksi LJK Otomatis")
    st.caption("Khusus Format LJK SMP YPI Pulogadung (40 Soal Pilihan Ganda - Tanda Silang)")

    def process_ljk_pil(pil_image, key_answers, threshold_val=110):
        gray_img = ImageOps.grayscale(pil_image)
        img_np = np.array(gray_img)
        
        h, w = img_np.shape
        
        # Area Estimasi Grid Pilihan Ganda (26% - 55% tinggi gambar)
        grid_top = int(h * 0.26)
        grid_bottom = int(h * 0.55)
        grid_left = int(w * 0.05)
        grid_right = int(w * 0.95)
        
        roi_pg = img_np[grid_top:grid_bottom, grid_left:grid_right]
        roi_h, roi_w = roi_pg.shape
        
        if roi_h == 0 or roi_w == 0:
            return 0, 0, []

        binary_roi = (roi_pg < threshold_val).astype(np.uint8)

        col_width = roi_w / 4
        row_height = roi_h / 10
        
        detected_answers = {}
        col_ranges = [
            range(1, 11),
            range(11, 21),
            range(21, 31),
            range(31, 41)
        ]
        options = ['A', 'B', 'C', 'D']
        
        for c_idx, q_range in enumerate(col_ranges):
            col_x_start = c_idx * col_width
            
            for r_idx, q_num in enumerate(q_range):
                row_y_start = r_idx * row_height
                options_x_start = col_x_start + (col_width * 0.25)
                options_width = col_width * 0.75
                opt_cell_width = options_width / 4
                
                max_pixels = 0
                selected_option = "-"
                
                for opt_idx, opt_label in enumerate(options):
                    x1 = int(options_x_start + (opt_idx * opt_cell_width) + (opt_cell_width * 0.15))
                    x2 = int(options_x_start + ((opt_idx + 1) * opt_cell_width) - (opt_cell_width * 0.15))
                    y1 = int(row_y_start + (row_height * 0.15))
                    y2 = int(row_y_start + row_height - (row_height * 0.15))
                    
                    x1, x2 = max(0, x1), min(roi_w, x2)
                    y1, y2 = max(0, y1), min(roi_h, y2)
                    
                    cell = binary_roi[y1:y2, x1:x2]
                    
                    if cell.size > 0:
                        pixel_count = np.sum(cell)
                        if pixel_count > 25 and pixel_count > max_pixels:
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
                "No Soal": q_num,
                "Jawaban Siswa": user_ans,
                "Kunci Jawaban": key_ans,
                "Status": "✅ Benar" if is_correct else "❌ Salah"
            })
            
        final_score = (score_correct / 40) * 100
        return final_score, score_correct, results

    # Sidebar / Pengaturan
    sidebar = st.sidebar
    sidebar.header("⚙️ Kunci Jawaban")

    key_dict = {}
    cols = sidebar.columns(2)
    for i in range(1, 41):
        col_target = cols[0] if i <= 20 else cols[1]
        key_dict[i] = col_target.selectbox(
            f"Soal {i}", 
            options=['A', 'B', 'C', 'D'], 
            index=0, 
            key=f"kunci_{i}"
        )

    threshold_sensitivity = sidebar.slider("Sensitivitas Deteksi Silang", 50, 200, 110, 5)

    uploaded_file = st.file_uploader("Unggah Foto LJK (Format JPG/PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_img, col_res = st.columns([1, 1])
        
        with col_img:
            st.subheader("🖼️ Lembar LJK Terunggah")
            st.image(image, use_container_width=True)
            
        score, correct_count, details = process_ljk_pil(image, key_dict, threshold_sensitivity)
        
        with col_res:
            st.subheader("📊 Hasil Koreksi")
            st.metric(label="Nilai Akhir (Skala 100)", value=f"{score:.1f}")
            st.write(f"**Jumlah Benar:** {correct_count} / 40 Soal")
            
            df_results = pd.DataFrame(details)
            st.dataframe(df_results, height=400, use_container_width=True)
    else:
        st.info("Silakan unggah foto lembar jawaban siswa untuk memulai koreksi.")

except Exception as e:
    st.error("Terjadi kesalahan teknis pada sistem:")
    st.code(traceback.format_exc())
