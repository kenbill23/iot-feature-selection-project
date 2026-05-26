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
st.sidebar.write("""
Aplikasi Machine Learning untuk klasifikasi kerentanan IoT menggunakan model terbaik hasil training.
""")

st.sidebar.success("Model berhasil dimuat")

# =========================
# Judul
# =========================
st.title("🎀 IoT Vulnerability Classification")
st.write("Prediksi kerentanan perangkat IoT menggunakan Machine Learning")

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🧠 Input Manual", "📂 Upload CSV"])

# ==================================================
# TAB INPUT MANUAL
# ==================================================
with tab1:

    st.subheader("Input Nilai Fitur Jaringan IoT")

    col1, col2 = st.columns(2)

    with col1:
        dur = st.number_input("dur", value=0.0004)
        protocol = st.number_input("Protocol", value=30)
        length = st.number_input("Length", value=54)
        source_host = st.number_input("Source Host", value=60674)

    with col2:
        destination_host = st.number_input("Destination Host", value=3472)
        sender_ip = st.number_input("Sender IP address", value=1)
        target_ip = st.number_input("Target IP address", value=1)

    if st.button("🎯 Prediksi Manual"):

        input_data = pd.DataFrame([{
            'dur': dur,
            'Protocol': protocol,
            'Length': length,
            'Source Host': source_host,
            'Destination Host': destination_host,
            'Sender IP address': sender_ip,
            'Target IP address': target_ip
        }])

        st.success("Input berhasil dimasukkan 🎉")
        st.dataframe(input_data)

# ==================================================
# TAB UPLOAD CSV
# ==================================================
with tab2:

    uploaded_file = st.file_uploader(
        "📂 Upload file CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            # Load CSV
            data = pd.read_csv(uploaded_file)

            st.subheader("📊 Dataset")
            st.dataframe(data.head())

            st.info(f"Jumlah data: {data.shape[0]} baris | {data.shape[1]} kolom")

            # Ambil fitur numerik
            data_numeric = data.select_dtypes(include=['int64', 'float64'])

            # Hapus kolom prediction jika ada
            if 'Prediction' in data_numeric.columns:
                data_numeric = data_numeric.drop(columns=['Prediction'])

            # Prediksi otomatis
            prediction = model.predict(data_numeric)

            # Tambahkan hasil
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