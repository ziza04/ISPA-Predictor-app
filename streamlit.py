import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Sistem prediksi ISPA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nama file untuk riwayat prediksi
HISTORY_FILE = "riwayat_prediksi.csv"

# Menu di sidebar
menu = st.sidebar.selectbox(
    "📂 Navigasi Menu",
    ["Form prediksi", "Riwayat prediksi"]
)

# Muat model dan encoder yang sudah dilatih
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_naive_bayes.joblib")
        label_encoder = joblib.load("label_encoder.joblib")
        return model, label_encoder
    except FileNotFoundError:
        st.error("❌ File model atau encoder tidak ditemukan. Pastikan 'model_naive_bayes.joblib' dan 'label_encoder.joblib' ada di direktori yang sama.")
        st.stop()

model, label_encoder = load_model()

# === Halaman 1: Form prediksi ===
if menu == "Form prediksi":
    st.title("🩺 Form prediksi ISPA")
    st.markdown("Isi informasi di bawah ini untuk mendapatkan prediksi awal ISPA Anda.")

    with st.form("form_prediksi"):
        st.subheader("🧍 Identitas Pengguna")
        col_id1, col_id2, col_id3 = st.columns(3)
        with col_id1:
            nama = st.text_input("Nama Lengkap", placeholder="Contoh: unknown")
        with col_id2:
            jk = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
        with col_id3:
            usia = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=0)

        st.markdown("---")
        st.subheader("📝 Gejala yang Dirasakan")
        st.markdown("Pilih atau sesuaikan tingkat gejala yang Anda alami.")

        col_gjl1, col_gjl2, col_gjl3 = st.columns(3)
        with col_gjl1:
            suhu = st.slider("Suhu Tubuh (°C)", 35.0, 42.0, 37.0, help="Suhu tubuh normal berkisar 36.5°C - 37.5°C.")
            nyeri_tenggorokan = st.slider("Skala Nyeri Tenggorokan (0-3)", 0, 3, 1, help="0: Tidak nyeri, 1: Ringan, 2: Sedang, 3: Parah.")
            nyeri_otot = st.slider("Nyeri Otot/Sendi (Skala 0-3)", 0, 3, 1, help="0: Tidak nyeri, 1: Ringan, 2: Sedang, 3: Parah.")
        with col_gjl2:
            batuk = st.slider("Frekuensi Batuk (per jam)", 0, 50, 10, help="Perkiraan berapa kali Anda batuk dalam satu jam.")
            sesak_napas = st.slider("Tingkat Sesak Napas (0-3)", 0, 3, 1, help="0: Tidak sesak, 1: Ringan, 2: Sedang, 3: Berat.")
            sakit_kepala = st.slider("Sakit Kepala (Skala 0-3)", 0, 3, 2, help="0: Tidak sakit, 1: Ringan, 2: Sedang, 3: Parah.")
        with col_gjl3:
            durasi = st.number_input("Durasi Sakit (hari)", min_value=0, max_value=30, value=3, help="Berapa lama gejala ini sudah Anda rasakan?")
            nafsu_makan = st.slider("Penurunan Nafsu Makan (Skala 0-3)", 0, 3, 2, help="0: Tidak ada penurunan, 1: Sedikit, 2: Sedang, 3: Drastis.")
            kontak = st.selectbox("Riwayat Kontak dengan Penderita ISPA?", ["Tidak", "Ya"], help="Pernahkah Anda berinteraksi dengan orang yang diprediksi ISPA?")

        st.markdown("---")
        submitted = st.form_submit_button("🔍 Dapatkan prediksi")

    if submitted:
        if not nama:
            st.warning("Mohon isi **Nama Lengkap** Anda untuk melanjutkan.")
        else:
            input_user = {
                'Suhu_Tubuh_C': suhu,
                'Frekuensi_Batuk_per_Jam': batuk,
                'Skala_Nyeri_Tenggorokan': nyeri_tenggorokan,
                'Tingkat_Sesak_Napas': sesak_napas,
                'Durasi_Sakit_Hari': durasi,
                'Nyeri_Otot_Sendi_Skala': nyeri_otot,
                'Sakit_Kepala_Skala': sakit_kepala,
                'Nafsu_Makan_Penurunan_Skala': nafsu_makan,
                'Usia_Tahun': usia,
                'Riwayat_Kontak_Positif': 1 if kontak == "Ya" else 0
            }

            input_df = pd.DataFrame([input_user])

            probabilities = model.predict_proba(input_df)[0]
            class_labels = label_encoder.classes_
            max_prob_index = np.argmax(probabilities)
            predicted_class = class_labels[max_prob_index]
            predicted_probability = probabilities[max_prob_index]

            st.markdown("---")
            st.subheader("📋 Hasil prediksi Anda")

            if predicted_class.lower() == 'tidak ispa':
                st.balloons()
                st.success(f"🥳 **Selamat, {nama}! Berdasarkan gejala yang Anda masukkan, kemungkinan besar Anda *TIDAK* mengalami ISPA.**")
                st.info(f"**Probabilitas:** **{predicted_probability*100:.0f}%** untuk **Tidak ISPA**.")
                st.write("Tetap jaga kesehatan dan kebersihan diri ya!")
                st.markdown("""
                **Saran Umum:** Tetap jaga pola hidup sehat (nutrisi, olahraga, istirahat cukup) dan selalu cuci tangan dengan sabun secara teratur.
                """)
            else:
                st.error(f"⚠️ **Perhatian, {nama}! Berdasarkan gejala Anda, kemungkinan besar Anda mengalami ISPA jenis: _{predicted_class}_**")
                st.info(f"**Probabilitas:** **{predicted_probability*100:.0f}%** untuk **{predicted_class}**.")
                st.markdown("**Probabilitas untuk kemungkinan lainnya:**")
                other_probs = {class_labels[i]: prob for i, prob in enumerate(probabilities) if i != max_prob_index and prob >= 0.01}
                if other_probs:
                    for label, prob in other_probs.items():
                        st.write(f"- {label}: {prob*100:.0f}%")
                else:
                    st.write("Tidak ada kemungkinan lain yang signifikan (>1%).")

                st.markdown("---")
                st.subheader("💡 Saran Penanganan Awal:")

                if predicted_class == 'Bronkitis':
                    st.markdown("""
                    **Konsultasi dokter sangat disarankan, terutama jika batuk berdahak parah atau sesak napas.** Istirahat, banyak minum air, hindari iritan (asap). Antibiotik jika bakteri.
                    """)
                elif predicted_class == 'Pneumonia':
                    st.markdown("""
                    **KONDISI INI SERIUS! SEGERA KE DOKTER/RUMAH SAKIT untuk penanganan medis intensif.** Jangan mengobati sendiri dan penting untuk istirahat penuh.
                    """)
                elif predicted_class == 'Faringitis':
                    st.markdown("""
                    **Jika gejala tidak membaik dalam beberapa hari, demam tinggi, atau nyeri sangat parah, segera konsultasi ke dokter.** Istirahat suara, minum hangat, lozenges.
                    """)
                elif predicted_class == 'Tonsilitis':
                    st.markdown("""
                    **Jika gejala tidak membaik dalam beberapa hari, demam tinggi, atau nyeri sangat parah, segera konsultasi ke dokter.** Istirahat, minum hangat, obat pereda nyeri. Antibiotik jika bakteri.
                    """)
                elif predicted_class == 'Rinitis':
                    st.markdown("""
                    **Jika gejala tidak membaik atau disertai demam tinggi, segera konsultasi ke dokter.** Istirahat, minum cukup air, bilas hidung dengan saline, antihistamin jika alergi.
                    """)
                else:
                    st.markdown("prediksi Anda adalah **" + predicted_class + "**. Disarankan untuk segera **berkonsultasi dengan tenaga medis** untuk prediksi dan penanganan lebih lanjut.")

            # Simpan hasil ke riwayat
            hasil = {
                'Nama': nama,
                'Jenis Kelamin': jk,
                'Usia': usia,
                'prediksi': predicted_class,
                'Probabilitas': round(predicted_probability, 2),
                'Waktu': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            riwayat_df = pd.DataFrame([hasil])
            if os.path.exists(HISTORY_FILE):
                riwayat_df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
            else:
                riwayat_df.to_csv(HISTORY_FILE, index=False)

# === Halaman 2: Riwayat prediksi ===
elif menu == "Riwayat prediksi":
    st.title("📁 Riwayat prediksi Anda")
    st.markdown("Lihat kembali semua riwayat prediksi yang pernah Anda lakukan melalui sistem ini.")

    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
        df_hist['Probabilitas'] = pd.to_numeric(df_hist['Probabilitas'], errors='coerce')
        df_hist.dropna(subset=['Probabilitas'], inplace=True)

        if not df_hist.empty:
            df_hist_sorted = df_hist.sort_values(by='Waktu', ascending=False)

            st.subheader("Semua Riwayat prediksi Pengguna:")
            st.dataframe(
                df_hist_sorted[[
                    'Waktu',
                    'Nama',
                    'Jenis Kelamin',
                    'Usia',
                    'prediksi',
                    'Probabilitas'
                ]].style.format({'Probabilitas': '{:.0%}'}),
                use_container_width=True
            )

            st.markdown("---")
        else:
            st.warning("⚠️ Belum ada data prediksi yang tersimpan. Silakan gunakan 'Form prediksi' terlebih dahulu.")
