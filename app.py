# app.py

import streamlit as st
import pandas as pd
import joblib

# ================================
# Load Model
# ================================

model = joblib.load("pipeline_terbaik.pkl")

# ================================
# Judul App
# ================================

st.title("IoT Vulnerability Classification")
st.write("Prediksi kerentanan perangkat IoT menggunakan Machine Learning")

# ================================
# Upload File CSV
# ================================

uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # ================================
    # Load Data
    # ================================

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(data.head(10))

    # ================================
    # Ambil fitur numerik
    # ================================

    data_numeric = data.select_dtypes(include=['int64', 'float64'])

    # ================================
    # Prediksi
    # ================================

    prediction = model.predict(data_numeric)

    # ================================
    # Hasil Prediksi
    # ================================

    st.subheader("Hasil Prediksi")

    data['Prediction'] = prediction

    # tampilkan sebagian data saja
    st.dataframe(data.head(100))

    # ================================
    # Download Hasil
    # ================================

    csv = data.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Hasil Prediksi",
        data=csv,
        file_name='hasil_prediksi.csv',
        mime='text/csv'
    )