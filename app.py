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
# Label Mapping
# =========================
label_mapping = {
    0: "ARPPoisoning",
    1: "Backdoor",
    2: "ICMPflood",
    3: "ICMPredirect",
    4: "Normal",
    5: "Password_crack",
    6: "Port_Scanning",
    7: "SQLInjection",
    8: "SYN_FLOOD",
    9: "Smurf",
    10: "UDP_flood",
    11: "Vulnerability_Scan"
}

# =========================
# Sidebar
# =========================
st.sidebar.title("🎀 IoT Vulnerability")

st.sidebar.success("✅ Model Loaded")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Deskripsi")

st.sidebar.info(
    "Sistem klasifikasi serangan IoT menggunakan model Machine Learning terbaik hasil training."
)

st.sidebar.markdown("---")

st.sidebar.subheader("🏆 Model Terbaik")

st.sidebar.markdown("""
**Random Forest Classifier**

**Best Score Cross Validation:** 95.69%

**Best Parameters**
- model__n_estimators = 10
- model__max_depth = 5
- feature_selection__estimator__n_estimators = 5
- feature_selection__estimator__max_depth = 5
""")

st.sidebar.markdown("---")

st.sidebar.subheader("👥 Kelompok")

st.sidebar.markdown("""
**Nabilla Wulandari** — 8020230008

**Fadillah Dwi Cahyanti** — 8020230023

**Revi Febrianti** — 8020230084
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
    "📂 Upload File CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # =========================
        # Load Dataset
        # =========================
        data = pd.read_csv(uploaded_file)

        display_data = data.copy()

        target_cols = [
            "Label",
            "Attack_Category",
            "Attack_sub_category"
        ]

        display_data = display_data.drop(
            columns=[
                col for col in target_cols
                if col in display_data.columns
            ],
            errors="ignore"
        )

        st.subheader("📊 Dataset")

        st.dataframe(
            display_data.head(),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            f"Jumlah data: {display_data.shape[0]} baris | {display_data.shape[1]} kolom"
        )

        # =========================
        # Data Prediksi
        # =========================
        data_pred = data.copy()

        data_pred = data_pred.drop(
            columns=[
                col for col in target_cols
                if col in data_pred.columns
            ],
            errors="ignore"
        )

        # =========================
        # Sesuaikan Fitur Model
        # =========================
        if hasattr(model, "feature_names_in_"):

            expected_features = list(model.feature_names_in_)

            missing_features = [
                col
                for col in expected_features
                if col not in data_pred.columns
            ]

            if missing_features:

                st.error(
                    f"Kolom fitur yang belum tersedia: {missing_features}"
                )

                st.stop()

            data_pred = data_pred[expected_features]

        # =========================
        # Prediksi
        # =========================
        prediction = model.predict(data_pred)

        prediction_label = [
            label_mapping.get(int(pred), "Unknown")
            for pred in prediction
        ]

        hasil = pd.DataFrame(
            {
                "Kode Prediksi": prediction,
                "Jenis Serangan": prediction_label
            }
        )

        st.success(
            "🎉 Prediksi berhasil dilakukan"
        )

        st.subheader(
            "📈 Hasil Prediksi"
        )

        st.dataframe(
            hasil,
            use_container_width=True,
            hide_index=True,
            height=350
        )

        # =========================
        # Download
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

        st.error(
            f"Terjadi kesalahan: {str(e)}"
        )