
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -----------------------------
# Load pipeline model
# -----------------------------
model_path = hf_hub_download(
    repo_id="Rajanan/model-predictive-engine-maintenance",
    filename="best_engine_failure_model.joblib",
    repo_type="model"
)

model = joblib.load(model_path)

# -----------------------------
# UI
# -----------------------------
st.title("Predictive Engine Maintenance System")

st.markdown("""
Predict whether an engine requires maintenance based on sensor inputs.
""")

# -----------------------------
# User Inputs
# -----------------------------
engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=800)

lub_oil_pressure = st.number_input("Lub Oil Pressure", min_value=0.0, max_value=10.0, value=3.0)

fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, max_value=25.0, value=6.0)

coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0, max_value=10.0, value=2.0)

lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=50.0, max_value=120.0, value=75.0)

coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=50.0, max_value=200.0, value=80.0)

# -----------------------------
# Create input DataFrame
# -----------------------------
input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])

# -----------------------------
# Prediction with threshold
# -----------------------------
if st.button("Predict Engine Condition"):

    # Probability
    prob = model.predict_proba(input_data)[0][1]

    # SAME threshold used in training
    threshold = 0.4

    prediction = 1 if prob >= threshold else 0

    st.write(f"Failure Probability: {prob:.2f}")

    if prediction == 1:
        st.error("Engine likely requires maintenance!")
    else:
        st.success("Engine operating normally")
