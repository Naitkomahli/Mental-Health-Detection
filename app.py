import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Judul dan Deskripsi Aplikasi
st.set_page_config(page_title="Prediksi Kesehatan Mental", page_icon="🧠")
st.title("🧠 Aplikasi Prediksi Kesehatan Mental")
st.write(
    "Aplikasi ini menggunakan model Machine Learning untuk memberikan prediksi awal "
    "tentang potensi depresi berdasarkan beberapa faktor gaya hidup dan demografis. "
    "Harap diingat, ini **bukan diagnosis medis**. Silakan berkonsultasi dengan "
    "profesional kesehatan untuk diagnosis yang akurat."
)
st.write("---")

# Fungsi untuk memuat model
# Menggunakan cache agar model tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model(model_path):
    """
    Memuat model machine learning dari file .pkl
    """
    try:
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        st.error(f"Error: File model tidak ditemukan di '{model_path}'. Pastikan file berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Memuat objek dari file pickle
loaded_object = load_model('simple_svm_model.pkl')
model = None

# Memeriksa apakah objek yang dimuat adalah dictionary dan mengekstrak modelnya
# Ini adalah praktik umum untuk menyimpan model bersama dengan metadata lainnya
if isinstance(loaded_object, dict):
    # Diasumsikan model disimpan dengan kunci 'model'
    model = loaded_object.get('model')
    if model is None:
        st.error("File .pkl adalah sebuah dictionary, tetapi tidak ditemukan kunci 'model' di dalamnya.")
else:
    # Jika bukan dictionary, diasumsikan objek tersebut adalah modelnya
    model = loaded_object


if model:
    # Menampilkan form untuk input pengguna
    st.header("Jawab Pertanyaan Berikut:")
    
    with st.form("prediction_form"):
        # Membuat 2 kolom untuk tata letak yang lebih rapi
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("1. Berapa usia Anda?", 15, 60, 25)
            work_study_hours = st.slider("2. Berapa jam Anda bekerja/belajar setiap hari?", 0, 16, 8)
            gender = st.selectbox("3. Apa jenis kelamin Anda?", ["Laki-laki", "Perempuan"])
            sleep_duration = st.selectbox(
                "4. Berapa lama durasi tidur Anda biasanya?",
                ["Kurang dari 5 jam", "5-6 jam", "7-8 jam", "Lebih dari 8 jam"]
            )

        with col2:
            financial_stress = st.slider("5. Bagaimana tingkat stres keuangan Anda? (1: Sangat Rendah, 5: Sangat Tinggi)", 1, 5, 3)
            family_history = st.radio("6. Apakah ada riwayat gangguan mental di keluarga Anda?", ["Ya", "Tidak"], index=1)
            suicidal_thoughts = st.radio("7. Apakah Anda pernah berpikir untuk mengakhiri hidup?", ["Ya", "Tidak"], index=1)
        
        # Tombol submit
        submit_button = st.form_submit_button(label="Dapatkan Prediksi")

    if submit_button:
        # --- Preprocessing Input Pengguna ---
        # Membuat dictionary untuk menampung data yang akan diubah menjadi DataFrame
        # Nama kolom harus SAMA PERSIS dengan yang digunakan saat training model
        
        # Daftar fitur yang diharapkan oleh model
        expected_features = [
            'Age', 'Work/Study Hours', 'Gender_Male', "Sleep Duration_'7-8 hours'",
            "Sleep Duration_'Less than 5 hours'", "Sleep Duration_'More than 8 hours'",
            'Sleep Duration_Others', 'Financial Stress_2.0', 'Financial Stress_3.0',
            'Financial Stress_4.0', 'Financial Stress_5.0', 'Financial Stress_?',
            'Family History of Mental Illness_Yes', 'Have you ever had suicidal thoughts ?_Yes'
        ]
        
        # Inisialisasi semua fitur dengan nilai 0
        input_data = {feature: 0 for feature in expected_features}

        # Mengisi nilai berdasarkan input pengguna
        input_data['Age'] = age
        input_data['Work/Study Hours'] = work_study_hours
        
        # Gender (One-Hot Encoding manual)
        if gender == 'Laki-laki':
            input_data['Gender_Male'] = 1
            
        # Durasi Tidur (One-Hot Encoding manual)
        if sleep_duration == '7-8 jam':
            input_data["Sleep Duration_'7-8 hours'"] = 1
        elif sleep_duration == 'Kurang dari 5 jam':
            input_data["Sleep Duration_'Less than 5 hours'"] = 1
        elif sleep_duration == 'Lebih dari 8 jam':
            input_data["Sleep Duration_'More than 8 hours'"] = 1
        else: # '5-6 jam' atau kategori lain akan masuk ke 'Others'
            input_data['Sleep Duration_Others'] = 1
            
        # Stres Keuangan (One-Hot Encoding manual)
        if financial_stress in [2, 3, 4, 5]:
            input_data[f'Financial Stress_{float(financial_stress)}'] = 1

        # Riwayat Keluarga
        if family_history == 'Ya':
            input_data['Family History of Mental Illness_Yes'] = 1
            
        # Pikiran Bunuh Diri
        if suicidal_thoughts == 'Ya':
            input_data['Have you ever had suicidal thoughts ?_Yes'] = 1

        # Membuat DataFrame dari input yang sudah diproses
        input_df = pd.DataFrame([input_data])
        # Memastikan urutan kolom sesuai dengan yang diharapkan model
        input_df = input_df[expected_features]

        # Melakukan prediksi
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Menampilkan hasil prediksi
        st.write("---")
        st.header("Hasil Prediksi Anda")

        if prediction[0] == 1:
            st.warning("**Potensi Mengalami Depresi**")
            st.write(
                f"Berdasarkan jawaban Anda, model memprediksi kemungkinan **{prediction_proba[0][1]*100:.2f}%** "
                "bahwa Anda mungkin mengalami gejala depresi. Kami sangat menyarankan untuk "
                "berkonsultasi dengan psikolog atau profesional kesehatan mental untuk mendapatkan "
                "evaluasi lebih lanjut."
            )
        else:
            st.success("**Potensi Tidak Mengalami Depresi**")
            st.write(
                f"Berdasarkan jawaban Anda, model memprediksi kemungkinan **{prediction_proba[0][0]*100:.2f}%** "
                "bahwa Anda tidak mengalami gejala depresi. Tetap jaga kesehatan mental Anda dengan "
                "gaya hidup sehat dan seimbang."
            )

        # Menampilkan disclaimer penting
        st.info(
            "**Penting:** Hasil ini adalah prediksi statistik dan **bukan pengganti diagnosis medis**. "
            "Kondisi kesehatan mental sangat kompleks dan hanya bisa didiagnosis oleh ahlinya."
        )

        with st.expander("Lihat detail input Anda"):
            st.write(input_df)

else:
    st.warning("Model tidak dapat dimuat. Aplikasi tidak dapat berjalan.")

