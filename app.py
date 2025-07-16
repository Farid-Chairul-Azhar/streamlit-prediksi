import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd 
import os
from tensorflow.keras.models import load_model

# ----------------------------
# Setup Aplikasi
# ----------------------------
st.set_page_config(page_title="Klasifikasi Sinyal EKG", layout="centered")
st.title("Deteksi Penyakit Jantung dari Sinyal EKG")
st.markdown("Unggah file sinyal EKG berdimensi **(1000, 12)** dalam format `.npy` untuk dianalisis.")

# ----------------------------
# Load Model & Aset Lainnya
# ----------------------------
model_path = "model_fold_3.keras"
scaler_path = "standard_scaler.joblib"
class_names_path = "class_names.npy"

@st.cache_resource
def load_artifacts():
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        class_names = np.load(class_names_path, allow_pickle=True)
        return model, scaler, class_names
    except Exception as e:
        st.error(f"Error saat memuat artefak: {e}")
        return None, None, None

model, scaler, class_names = load_artifacts()

# ----------------------------
# Upload File Sinyal EKG
# ----------------------------
uploaded_file = st.file_uploader("Unggah file .npy", type="npy")

# ----------------------------
# Form Tambahan: Metadata Pasien
# ----------------------------
with st.expander("Tambahkan Informasi Pasien (Opsional)", expanded=True):
    nama_pasien = st.text_input("Nama Pasien", value="Tidak diketahui")
    umur_pasien = st.number_input("Usia Pasien", min_value=0, max_value=120, value=0)
    jenis_kelamin = st.selectbox("Jenis Kelamin", options=["Tidak diketahui", "Laki-laki", "Perempuan"])

# ----------------------------
# Proses Prediksi
# ----------------------------
if uploaded_file is not None:
    try:
        signal = np.load(uploaded_file, allow_pickle=True)

        if signal.shape != (1000, 12):
            st.error(f"Bentuk sinyal tidak sesuai. Diperlukan (1000, 12), namun ditemukan {signal.shape}.")
        else:
            signal_scaled = scaler.transform(signal)
            signal_ready = signal_scaled.reshape(1, 1000, 12)

            prediction = model.predict(signal_ready)
            pred_label = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            # ------------------------
            # Tampilkan Informasi Pasien
            # ------------------------
            st.subheader("Informasi Pasien")
            st.write(f"**Nama**          : {nama_pasien}")
            st.write(f"**Usia**          : {umur_pasien} tahun")
            st.write(f"**Jenis Kelamin** : {jenis_kelamin}")

            # ------------------------
            # Hasil Prediksi
            # ------------------------
            st.subheader("Hasil Prediksi")
            st.success(f"Model memprediksi: **{pred_label}**")
            st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2%}")

            # ------------------------
            # Probabilitas Semua Kelas
            # ------------------------
            df_probs = pd.DataFrame({
                'Kelas': class_names,
                'Probabilitas': prediction[0]
            })
            st.bar_chart(df_probs.set_index('Kelas'))

            # ------------------------
            # Visualisasi Lead I
            # ------------------------
            st.subheader("Visualisasi Sinyal (Lead I)")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(signal[:, 0], color='blue')
            ax.set_title("Lead I")
            ax.set_xlabel("Titik Waktu")
            ax.set_ylabel("Amplitudo")
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
