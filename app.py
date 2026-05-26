import streamlit as st
import pandas as pd
import joblib
# =========================
# Upload CSV
# =========================
uploaded_file = st.file_uploader(
    "📂 Upload file CSV",
    type=["csv"]
)

# =========================
# Prediksi Otomatis
# =========================
if uploaded_file is not None:

    try:
        # Load data
        data = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset")
        st.dataframe(data.head())

        st.info(f"Jumlah data: {data.shape[0]} baris | {data.shape[1]} kolom")

        # Ambil fitur numerik
        data_numeric = data.select_dtypes(include=['int64', 'float64'])

        # Hapus kolom prediction kalau ada
        if 'Prediction' in data_numeric.columns:
            data_numeric = data_numeric.drop(columns=['Prediction'])

        # Prediksi otomatis
        prediction = model.predict(data_numeric)

        # Tambahkan hasil prediksi
        data['Prediction'] = prediction

        st.success("🎉 Prediksi berhasil dilakukan")

        st.subheader("📈 Hasil Prediksi")
        st.dataframe(data.head())

        # Download hasil
        csv = data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download Hasil Prediksi",
            data=csv,
            file_name='hasil_prediksi.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")