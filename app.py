import streamlit as st
import pandas as pd
import joblib

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="IoT Vulnerability Classification",
    page_icon="🎀",
    layout="wide"
)

# =========================
# Styling Pink Theme
# =========================
st.markdown("""
<style>

.stApp {
    background-color: #ffc0cb;
}

h1, h2, h3, p, label {
    color: black !important;
}

[data-testid="stSidebar"] {
    background-color: #ffb6c1;
}

.stButton>button {
    background-color: #ff69b4;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stDownloadButton>button {
    background-color: #ff69b4;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

div[data-testid="stFileUploader"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("pipeline_terbaik.pkl")

model = load_model()

# =========================
# Sidebar
# =========================
st.sidebar.title("🎀 About App")
st.sidebar.write(
    "Aplikasi Machine Learning untuk klasifikasi kerentanan IoT menggunakan model terbaik hasil training."
)

st.sidebar.success("Model berhasil dimuat")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Fitur")
st.sidebar.write("""
✅ Upload Dataset CSV

✅ Prediksi Kerentanan IoT

✅ Download Hasil Prediksi

✅ Pipeline Machine Learning
""")

st.sidebar.markdown("---")

st.sidebar.subheader("👩‍💻 Kelompok")
st.sidebar.write("""
Nabilla Wulan

Fadilla

Revi
""")

# =========================
# Judul
# =========================
st.title("🎀 IoT Vulnerability Classification")

st.write(
    "Upload dataset CSV untuk melakukan prediksi kerentanan perangkat IoT."
)

st.markdown("---")

# =========================
# Upload CSV
# =========================
uploaded_file = st.file_uploader(
    "📂 Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # Load Dataset
        data = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset")
        st.dataframe(data.head())

        st.info(
            f"Jumlah data: {data.shape[0]} baris | {data.shape[1]} kolom"
        )

        # =========================
        # Menyesuaikan fitur training
        # =========================

        if hasattr(model, "feature_names_in_"):

            expected_features = model.feature_names_in_

            missing_features = [
                col for col in expected_features
                if col not in data.columns
            ]

            if missing_features:
                st.error(
                    f"Kolom yang dibutuhkan model tidak ditemukan: {missing_features}"
                )
                st.stop()

            data_pred = data[expected_features]

        else:
            data_pred = data.copy()

        # =========================
        # Prediksi
        # =========================

        prediction = model.predict(data_pred)

        # Hanya tampilkan hasil prediksi
        hasil = pd.DataFrame({
            "Prediction": prediction
        })

        st.success("🎉 Prediksi berhasil dilakukan")

        st.subheader("📈 Hasil Prediksi")
        st.dataframe(hasil)

        # =========================
        # Download Hasil
        # =========================

        csv = hasil.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")