import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="IoT Vulnerability Classification",
    page_icon="🎀",
    layout="wide"
)

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
}

.stDownloadButton>button {
    background-color: #ff69b4;
    color: white;
    border-radius: 12px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("pipeline_terbaik.pkl")

model = load_model()

st.sidebar.title("🎀 About App")
st.sidebar.write(
    "Aplikasi klasifikasi kerentanan IoT menggunakan model terbaik."
)

st.title("🎀 IoT Vulnerability Classification")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        st.subheader("Dataset")
        st.dataframe(data.head())

        data_pred = data.copy()

        drop_cols = [
            "Prediction",
            "Attack_sub_category",
            "Attack_Category"
        ]

        for col in drop_cols:
            if col in data_pred.columns:
                data_pred.drop(columns=col, inplace=True)

        prediction = model.predict(data_pred)

        hasil = data.copy()
        hasil["Prediction"] = prediction

        st.success("Prediksi berhasil")

        st.subheader("Hasil Prediksi")
        st.dataframe(hasil.head())

        csv = hasil.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Hasil",
            csv,
            "hasil_prediksi.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")