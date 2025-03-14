import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import base64
from io import BytesIO

# Load Model & Scaler
with open("xgboost_heart_disease.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="AI Heart Disease Prediction", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    body {background-color: #f8f9fa;}
    .main {background-color: white; padding: 20px; border-radius: 10px;}
    h1 {color: #dc3545;}
    </style>
    """, unsafe_allow_html=True
)

st.title("💖 AI-Powered Heart Disease Prediction")

# Input Features
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 50)
    gender = st.radio("Gender", ["Male", "Female"])
    systolic_bp = st.slider("Systolic BP", 80, 200, 120)
    diastolic_bp = st.slider("Diastolic BP", 50, 130, 80)
    cholesterol = st.selectbox("Cholesterol Level", sorted(["Normal", "Above Normal", "High"]))

with col2:
    glucose = st.selectbox("Glucose Level", sorted(["Normal", "Above Normal", "High"]))
    smoker = st.radio("Do you smoke?", ["No", "Yes"])
    alcohol = st.radio("Do you consume alcohol?", ["No", "Yes"])
    physical_activity = st.radio("Are you physically active?", ["No", "Yes"])
    height = st.slider("Height (cm)", 140, 200, 170)
    weight = st.slider("Weight (kg)", 40, 150, 70)

# Convert categorical inputs to numeric values
gender = 1 if gender == "Male" else 0
cholesterol = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]
glucose = {"Normal": 1, "Above Normal": 2, "High": 3}[glucose]
smoker = 1 if smoker == "Yes" else 0
alcohol = 1 if alcohol == "Yes" else 0
physical_activity = 1 if physical_activity == "Yes" else 0

# Feature Engineering
bmi = weight / ((height / 100) ** 2)
bp_ratio = systolic_bp / diastolic_bp

# Create Feature DataFrame
input_data = pd.DataFrame([[age, gender, systolic_bp, diastolic_bp, cholesterol, glucose, smoker, alcohol, physical_activity, bmi, bp_ratio]],
                          columns=["Age", "Gender", "Systolic_BP", "Diastolic_BP", "Cholesterol", "Glucose", "Smoker", "Alcohol", "Physical_Activity", "BMI", "BP_Ratio"])

# Scale Data
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)[0]
prediction_proba = model.predict_proba(input_data_scaled)[:, 1][0]

# Show Results
st.subheader("🩺 Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Disease Detected! (Risk: {prediction_proba:.2%})")
    disease_name = "Possible Cardiovascular Disease"
    st.write(f"📝 **Detected Disease:** {disease_name}")
else:
    st.success(f"✅ No Immediate Heart Disease Risk Detected! (Risk: {prediction_proba:.2%})")

# SHAP Explanation
st.subheader("🔍 Model Explainability (SHAP Analysis)")
explainer = shap.Explainer(model)
shap_values = explainer(input_data_scaled)
st.pyplot(shap.plots.waterfall(shap_values[0]))

# PDF Report Generation
def generate_pdf_report():
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Heart Disease Prediction Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}", ln=True)
    pdf.cell(200, 10, f"Risk Probability: {prediction_proba:.2%}", ln=True)

    pdf.cell(200, 10, "Feature Inputs:", ln=True)
    for col, val in zip(input_data.columns, input_data.values[0]):
        pdf.cell(200, 10, f"{col}: {val}", ln=True)

    pdf.output("Heart_Disease_Report.pdf")
    return "Heart_Disease_Report.pdf"

if st.button("📥 Download Report"):
    report_path = generate_pdf_report()
    with open(report_path, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
    href = f'<a href="data:file/pdf;base64,{b64}" download="Heart_Disease_Report.pdf">Click Here to Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# File Upload for Batch Prediction
st.subheader("📤 Upload Patient Data (CSV/PDF)")
uploaded_file = st.file_uploader("Upload a patient dataset for automatic predictions", type=["csv", "pdf"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        df_scaled = scaler.transform(df)
        predictions = model.predict(df_scaled)
        df["Prediction"] = predictions
        st.write(df)
        st.download_button("📥 Download Predictions", df.to_csv(index=False), file_name="Predictions.csv", mime="text/csv")
    else:
        st.warning("PDF Processing Coming Soon!")

# Footer
st.markdown("Developed by **Ankush** with ❤️")