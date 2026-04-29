import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# ===== LOAD MODEL =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "model", "encoders.pkl"))

# ===== SAFE ENCODER =====
def safe_encode(col, value):
    le = encoders[col]
    return le.transform([value])[0] if value in le.classes_ else 0

# ===== RESET FUNCTION =====
def reset():
    st.session_state.clear()

# ===== PAGE CONFIG =====
st.set_page_config(page_title="❤️ AI Heart Risk Analyzer", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
                url("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Remove white header */
[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: rgba(20, 20, 20, 0.9);
}
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.title("❤️ AI Heart Risk Analyzer")
st.divider()

# ===== SIDEBAR =====
st.sidebar.markdown("## ⚙️ Patient Input")

age = st.sidebar.slider("Age", 20, 80, 40)
sex = st.sidebar.selectbox("Sex", list(encoders['sex'].classes_))
cp = st.sidebar.selectbox("Chest Pain", list(encoders['cp'].classes_))
trestbps = st.sidebar.slider("Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Sugar", list(encoders['fbs'].classes_))
restecg = st.sidebar.selectbox("ECG", list(encoders['restecg'].classes_))
thalch = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Angina", list(encoders['exang'].classes_))
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)

# ===== ENCODE =====
sex = safe_encode('sex', sex)
cp = safe_encode('cp', cp)
restecg = safe_encode('restecg', restecg)
fbs = safe_encode('fbs', fbs)
exang = safe_encode('exang', exang)

# ===== BUTTONS =====
col_btn1, col_btn2 = st.columns([3,1])

with col_btn1:
    analyze = st.button("🚀 Analyze Risk")

with col_btn2:
    st.button("🔄 Reset", on_click=reset)

# ===== MAIN =====
if analyze:

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak
    }])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)

    # ===== RISK =====
    with col1:
        st.subheader("📊 Risk Level")
        st.progress(min(int(prob * 100), 100))

        if prob > 0.7:
            color = "#ff4b4b"
            text = "HIGH RISK ⚠️"
        elif prob > 0.4:
            color = "#ffa500"
            text = "MODERATE RISK ⚡"
        else:
            color = "#00c853"
            text = "LOW RISK ✅"

        st.markdown(f"""
        <div style="padding:20px; border-radius:10px; background:{color}; color:white; text-align:center;">
        {text}<br>{prob*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

        # ✅ 1. Confidence
        st.info(f"Model confidence: {prob*100:.2f}%")

    # ===== SNAPSHOT =====
    with col2:
        st.subheader("🧍 Patient Snapshot")
        st.write(f"Age: {age}")
        st.write(f"BP: {trestbps}")
        st.write(f"Cholesterol: {chol}")
        st.write(f"Heart Rate: {thalch}")

    st.divider()

    # ===== RESULT =====
    st.subheader("🧠 AI Interpretation")

    if prediction == 1:
        st.error("⚠️ High probability of heart disease detected")
    else:
        st.success("✅ No significant risk detected")

    # ✅ 2. Recommendation System
    if prob > 0.7:
        st.warning("Recommendation: Consult a cardiologist immediately.")
    elif prob > 0.4:
        st.info("Recommendation: Maintain healthy lifestyle and monitor regularly.")
    else:
        st.success("Recommendation: Keep maintaining a healthy lifestyle.")

    # ===== FEATURE IMPORTANCE =====
    if hasattr(model, "feature_importances_"):
        st.subheader("🔥 Key Risk Factors")

        features = ["age","sex","cp","trestbps","chol","fbs","restecg","thalch","exang","oldpeak"]

        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feat_df.set_index("Feature"))

    st.divider()
    st.caption("⚠️ Educational use only, not medical advice.")