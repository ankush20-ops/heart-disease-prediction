import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF
import fitz  # PyMuPDF for PDF reading
import os  # For path handling

# Ensure paths are correct
import os
import pickle

# Correct path for Streamlit Cloud
MODEL_PATH = "xgboost_heart_disease.pkl"

# Load Model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Generate Data Image (Instead of 'heart_banner.jpg')
def generate_data_image():
    plt.figure(figsize=(10, 4))
    categories = ['Age', 'BP', 'Cholesterol', 'Glucose', 'Activity']
    values = [50, 120, 1, 1, 1]
    plt.bar(categories, values, color='#d62828')
    plt.title('Key Health Metrics for Heart Disease Prevention')
    plt.savefig("generated_banner.png")  # Save as temporary image
    return "generated_banner.png"

# Heart Disease Types Mapping
disease_types = {
    0: "No Heart Disease",
    1: "Coronary Artery Disease",
    2: "Arrhythmia",
    3: "Heart Valve Disease",
    4: "Cardiomyopathy"
}

# UI Enhancements
st.markdown("<h1 style='text-align: center;'>🫀 Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.image(generate_data_image(), use_container_width=True)  # Display generated data image
st.markdown(
    """
    <style>
        .stApp { background-color: #f5f5f5; }
        h1 { color: #d62828; }
        .sidebar .sidebar-content { background-color: #eae2b7; }
    </style>
    """, unsafe_allow_html=True
)

# Prediction Function
def predict_heart_disease(features):
    prediction = model.predict([features])[0]
    return prediction

# SHAP Explanation
def explain_prediction(features):
    explainer = shap.Explainer(model)
    shap_values = explainer(np.array([features]))
    
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0], ax=ax)
    st.pyplot(fig)

# PDF Report Generation
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, '🫀 Heart Disease Prediction Report', ln=True, align='C')
        self.ln(10)

    def add_prediction_details(self, data, prediction, suggestions):
        self.set_font('Arial', '', 12)
        for key, value in data.items():
            self.cell(0, 10, f"{key}: {value}", ln=True)
        self.ln(5)
        self.cell(0, 10, f"Prediction: {disease_types[prediction]}", ln=True)
        self.ln(5)
        self.multi_cell(0, 10, f"Suggestions: {suggestions}")
        self.ln(10)

# Generate Report
def generate_pdf_report(data, prediction, suggestions):
    pdf = PDFReport()
    pdf.add_page()
    pdf.add_prediction_details(data, prediction, suggestions)

    report_path = "Heart_Disease_Report.pdf"
    pdf.output(report_path)
    return report_path

# Health Suggestions
def get_health_suggestions(prediction):
    if prediction == 0:
        return "✅ Maintain a healthy lifestyle. Exercise regularly, eat a balanced diet, and avoid smoking or excessive alcohol."
    else:
        return (
            "❗ Important Steps: \n"
            "- Consult your doctor immediately.\n"
            "- Follow prescribed medications and treatments.\n"
            "- Maintain a heart-healthy diet.\n"
            "- Monitor blood pressure and cholesterol regularly."
        )

# PDF Data Extraction
def extract_data_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ''
    for page in doc:
        text += page.get_text()

    extracted_data = {}
    for line in text.split('\n'):
        if ':' in line:
            key, value = map(str.strip, line.split(':', 1))
            extracted_data[key] = value
    return extracted_data

# Input Form
st.sidebar.header("Enter Patient Data or Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload Patient Data (PDF)", type=["pdf"])

if uploaded_pdf:
    extracted_data = extract_data_from_pdf(uploaded_pdf)
    st.sidebar.write("Extracted Data:", extracted_data)
    patient_data = list(map(float, extracted_data.values()))
else:
    patient_data = [
        st.sidebar.number_input("Age", min_value=1, max_value=120, value=50),
        st.sidebar.selectbox("Gender", [0, 1]),
        st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=165),
        st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70),
        st.sidebar.number_input("Systolic BP", min_value=80, max_value=200, value=120),
        st.sidebar.number_input("Diastolic BP", min_value=40, max_value=130, value=80),
        st.sidebar.selectbox("Cholesterol Level", [1, 2, 3]),
        st.sidebar.selectbox("Glucose Level", [1, 2, 3]),
        st.sidebar.selectbox("Smoker?", [0, 1]),
        st.sidebar.selectbox("Alcohol Intake?", [0, 1]),
        st.sidebar.selectbox("Physical Activity?", [0, 1]),
    ]

# Prediction & Results
if st.sidebar.button("Predict"):
    prediction = predict_heart_disease(patient_data)
    suggestions = get_health_suggestions(prediction)

    st.success(f"🧬 Prediction: {disease_types.get(prediction, 'Unknown')}")
    st.info(f"💬 Suggestions: {suggestions}")

    # SHAP Explanation
    st.subheader("🔍 Feature Importance Analysis")
    explain_prediction(patient_data)

    # PDF Report Download
    report_path = generate_pdf_report(
        dict(zip(['Age', 'Gender', 'Height', 'Weight', 'Systolic BP', 'Diastolic BP',
                  'Cholesterol Level', 'Glucose Level', 'Smoker?', 'Alcohol Intake?', 'Physical Activity?'],
                 patient_data)),
        prediction,
        suggestions
    )

    with open(report_path, "rb") as file:
        st.download_button(
            label="📄 Download Report",
            data=file,
            file_name="Heart_Disease_Report.pdf",
            mime="application/pdf"
        )
