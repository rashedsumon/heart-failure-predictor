import os
import streamlit as st
import pandas as pd
import joblib
from data_loader import load_heart_data
from model import train_and_save_model

# Page config configuration
st.set_page_config(
    page_title="Heart Failure Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Failure Clinical Risk Predictor")
st.write("""
This interactive AI assistant uses a Random Forest machine learning model to estimate 
the risk of a mortality event during a patient's follow-up period based on clinical parameters.
""")

# --- Model Loading / Caching ---
MODEL_PATH = "heart_failure_model.joblib"

@st.cache_resource
def get_trained_model():
    """Ensures the model is ready. If not found, trains it on-the-fly."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Initializing Model & Downloading Kaggle Datasets... Please wait..."):
            return train_and_save_model()
    return joblib.load(MODEL_PATH)

# Instantiate the model
model = get_trained_model()

# --- Layout: Sidebar Inputs ---
st.sidebar.header("📋 Patient Clinical Metrics")

st.sidebar.subheader("Demographics & Habits")
age = st.sidebar.slider("Age", min_value=1, max_value=110, value=60, step=1)
sex = st.sidebar.selectbox("Biological Sex", options=["Female", "Male"], index=1)
smoking = st.sidebar.selectbox("Smoking Status", options=["No", "Yes"], index=0)

st.sidebar.subheader("Medical History")
anaemia = st.sidebar.selectbox("Anaemia (Low Red Blood Cells)", options=["No", "Yes"], index=0)
diabetes = st.sidebar.selectbox("Diabetes Diagnosis", options=["No", "Yes"], index=0)
high_blood_pressure = st.sidebar.selectbox("Hypertension (High BP)", options=["No", "Yes"], index=1)

st.sidebar.subheader("Lab Diagnostics & Vitals")
ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", min_value=10, max_value=80, value=35, step=1)
serum_creatinine = st.sidebar.slider("Serum Creatinine (mg/dL)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
serum_sodium = st.sidebar.slider("Serum Sodium (mEq/L)", min_value=100, max_value=150, value=135, step=1)
platelets = st.sidebar.number_input("Platelet Count (cells/mcL)", min_value=20000, max_value=900000, value=250000, step=5000)
creatinine_phosphokinase = st.sidebar.number_input("CPK Enzyme Level (mcg/L)", min_value=20, max_value=8000, value=250, step=10)

st.sidebar.subheader("Timeline")
time = st.sidebar.slider("Follow-up Window Period (Days)", min_value=1, max_value=300, value=45, step=1)

# --- Process User Inputs ---
# Convert UI labels back to 1s and 0s mapped to the model's training columns
input_data = pd.DataFrame([{
    'age': float(age),
    'anaemia': 1 if anaemia == "Yes" else 0,
    'creatinine_phosphokinase': int(creatinine_phosphokinase),
    'diabetes': 1 if diabetes == "Yes" else 0,
    'ejection_fraction': int(ejection_fraction),
    'high_blood_pressure': 1 if high_blood_pressure == "Yes" else 0,
    'platelets': float(platelets),
    'serum_creatinine': float(serum_creatinine),
    'serum_sodium': int(serum_sodium),
    'sex': 1 if sex == "Male" else 0,
    'smoking': 1 if smoking == "Yes" else 0,
    'time': int(time)
}])

# --- Main Page Dashboard layout ---
col1, col2 = pd.DataFrame().columns, pd.DataFrame().columns # Placeholder check
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Submitted Patient Parameters")
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.subheader("Prediction Evaluation")
    
    # Calculate Prediction & Probabilities
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    risk_percentage = probabilities[1] * 100
    
    # Display visually structured metrics blocks
    if prediction == 1:
        st.error(f"⚠️ **High Risk Group Detected**")
        st.metric(label="Mortality Risk Probability", value=f"{risk_percentage:.1f}%")
        st.write("The model indicates an elevated risk profile within this tracking window. Close clinical monitoring is advised.")
    else:
        st.success(f"✅ **Low Risk Group Detected**")
        st.metric(label="Mortality Risk Probability", value=f"{risk_percentage:.1f}%")
        st.write("The patient displays parameters that historical records correlate with high survival tracking probabilities.")

# --- Footnote Section ---
st.divider()
st.caption("Disclaimer: This tool is developed strictly for educational demonstration purposes based on Kaggle datasets and should not replace professional medical evaluations.")