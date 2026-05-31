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
        # Load dataset
        data = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset")
        st.dataframe(data.head())

        st.info(
            f"Jumlah data: {data.shape[0]} baris | {data.shape[1]} kolom"
        )

        # Simpan data asli untuk ditampilkan
        data_pred = data.copy()

        # Hapus kolom yang bukan fitur model
        drop_cols = [
            "Prediction",
            "Attack_sub_category",
            "Attack_Category"
        ]

        for col in drop_cols:
            if col in data_pred.columns:
                data_pred = data_pred.drop(columns=[col])

        # Prediksi menggunakan pipeline
        prediction = model.predict(data_pred)

        # Tambahkan hasil prediksi ke data asli
        hasil = data.copy()
        hasil["Prediction"] = prediction

        st.success("🎉 Prediksi berhasil dilakukan")

        st.subheader("📈 Hasil Prediksi")
        st.dataframe(hasil.head())

        # Download hasil
        csv = hasil.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")