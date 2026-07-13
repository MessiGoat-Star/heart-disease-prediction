import streamlit as st
import pandas as pd
import joblib

# =========================================================
# Heart Disease Prediction - Streamlit App (Bonus Deployment)
# Muat model hasil training dari notebook (best_model.pkl)
# =========================================================

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀")

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

st.title("🫀 Heart Disease Prediction")
st.write(
    "Aplikasi ini membantu skrining awal risiko penyakit jantung "
    "berdasarkan 13 variabel klinis pasien."
)

st.header("Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    cp = st.selectbox("Chest pain type", options=[1, 2, 3, 4])
    bp = st.number_input("BP (Resting Blood Pressure)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
    fbs = st.selectbox("FBS over 120", options=[0, 1])
    ekg = st.selectbox("EKG results", options=[0, 1, 2])

with col2:
    max_hr = st.number_input("Max HR", min_value=50, max_value=250, value=150)
    ex_ang = st.selectbox("Exercise angina", options=[0, 1])
    oldpeak = st.number_input("ST depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST", options=[1, 2, 3])
    vessels = st.selectbox("Number of vessels fluro", options=[0, 1, 2, 3])
    thal = st.selectbox("Thallium", options=[3, 6, 7])

input_data = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "Chest pain type": cp,
    "BP": bp,
    "Cholesterol": chol,
    "FBS over 120": fbs,
    "EKG results": ekg,
    "Max HR": max_hr,
    "Exercise angina": ex_ang,
    "ST depression": oldpeak,
    "Slope of ST": slope,
    "Number of vessels fluro": vessels,
    "Thallium": thal,
}])

if st.button("Prediksi"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    if pred == 1:
        st.error(f"⚠️ Pasien terindikasi **Presence** (berisiko penyakit jantung).")
    else:
        st.success(f"✅ Pasien terindikasi **Absence** (tidak berisiko penyakit jantung).")

    if proba is not None:
        st.write(f"Probabilitas risiko penyakit jantung: **{proba:.2%}**")

st.caption(
    "Catatan: hasil prediksi ini hanya untuk skrining awal dan bukan pengganti "
    "diagnosis medis profesional."
)
