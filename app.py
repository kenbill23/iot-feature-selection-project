import streamlit as st
import pandas as pd
import joblib

# =========================
# Load Model
# =========================
model = joblib.load("pipeline_terbaik.pkl")

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="IoT Vulnerability Classification",
    layout="wide"
)

# =========================
# Styling Pink UI
# =========================
st.markdown("""
    <style>
    .stApp {
        background-color: #ffc0cb;
    }

    h1, h2, h3, p, label {
        color: black;
    }

    .stButton>button {
        background-color: #ff69b4;
        color: white;
        border-radius: 10px;
        border: none;
    }

    .stDownloadButton>button {
        background-color: #ff69b4;
        color: white;
        border-radius: 10px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Judul
# =========================
st.title("IoT Vulnerability Classification")
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
        dur = st.number_input("dur", value=0.0)
        protocol = st.number_input("Protocol", value=0.0)
        length = st.number_input("Length", value=0.0)
        source_host = st.number_input("Source Host", value=0.0)

    with col2:
        destination_host = st.number_input("Destination Host", value=0.0)
        sender_ip = st.number_input("Sender IP address", value=0.0)
        target_ip = st.number_input("Target IP address", value=0.0)

    if st.button("Prediksi Manual"):

        input_data = pd.DataFrame([{
            'dur': dur,
            'Protocol': protocol,
            'Length': length,
            'Source Host': source_host,
            'Destination Host': destination_host,
            'Sender IP address': sender_ip,
            'Target IP address': target_ip
        }])

        prediction = model.predict(input_data)

        st.success(f"Hasil Prediksi: {prediction[0]}")

# ==================================================
# TAB UPLOAD CSV
# ==================================================
with tab2:

    uploaded_file = st.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.subheader("Dataset")
        st.dataframe(data.head())

        # Ambil fitur numerik
        data_numeric = data.select_dtypes(include=['int64', 'float64'])

        # Prediksi
        prediction = model.predict(data_numeric)

        # Tambahkan hasil
        data['Prediction'] = prediction

        st.subheader("Hasil Prediksi")
        st.dataframe(data.head())

        # Download hasil
        csv = data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Hasil Prediksi",
            data=csv,
            file_name='hasil_prediksi.csv',
            mime='text/csv'
        )