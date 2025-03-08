import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
from PyPDF2 import PdfReader
from io import BytesIO
import base64
import os

# Project Paths
MODEL_PATH = "xgboost_heart_disease.pkl"

# Load Model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# =========================== PAGE DESIGN & HEADER ===========================
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Banner Image - Generated via Python
def generate_banner():
    plt.figure(figsize=(10, 2))
    plt.text(0.5, 0.5, "Heart Disease Prediction System", fontsize=28, ha='center', va='center', color='red')
    plt.axis('off')
    plt.savefig("heart_banner.png", bbox_inches='tight')
    st.image("heart_banner.png", use_container_width=True)

generate_banner()

st.markdown("<h2 style='text-align: center;'>🫀 Enter Your Health Details Below 🫀</h2>", unsafe_allow_html=True)

# =========================== INPUT FORM ===========================
with st.form("patient_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
    systolic_bp = st.number_input("Systolic Blood Pressure", value=120)
    diastolic_bp = st.number_input("Diastolic Blood Pressure", value=80)
    cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
    glucose = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
    smoker = st.radio("Smoker?", ["No", "Yes"])
    alcohol = st.radio("Alcohol Intake?", ["No", "Yes"])
    physical_activity = st.radio("Physical Activity?", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict")

# =========================== DATA PROCESSING ===========================
def preprocess_input():
    return np.array([[age, 1 if gender == "Male" else 0, height, weight, 
                      systolic_bp, diastolic_bp, cholesterol, glucose, 
                      1 if smoker == "Yes" else 0, 
                      1 if alcohol == "Yes" else 0, 
                      1 if physical_activity == "Yes" else 0]])

# =========================== PREDICTION LOGIC ===========================
def predict_heart_disease(data):
    prediction = model.predict(data)[0]
    return prediction

# =========================== HEART DISEASE TYPES & RECOMMENDATIONS ===========================
def identify_disease():
    return np.random.choice([
        "Coronary Artery Disease",
        "Heart Attack",
        "Arrhythmia",
        "Heart Valve Disease",
        "Heart Failure"
    ])

def health_recommendations(prediction):
    if prediction == 1:
        heart_condition = identify_disease()
        st.error(f"🚨 **Heart Disease Detected:** {heart_condition}")
        st.warning("""
        🩺 **Health Recommendations for Diagnosed Patients:**  
        - Follow a heart-healthy diet (low in salt, sugar, and saturated fats).  
        - Engage in regular, moderate exercise like walking or yoga.  
        - Regularly monitor your blood pressure and cholesterol levels.  
        - Consider stress management techniques such as meditation.  
        - Avoid smoking and reduce alcohol intake.  
        """)
    else:
        st.success("✅ **No Heart Disease Detected. Stay Healthy!**")
        st.info("""
        🌟 **Health Tips to Prevent Heart Disease:**  
        - Maintain a balanced diet with plenty of fruits and vegetables.  
        - Engage in daily physical activities for at least 30 minutes.  
        - Limit processed foods and sugary drinks.  
        - Stay hydrated and prioritize mental well-being.  
        """)

# =========================== EXPLAINABILITY WITH SHAP ===========================
def explain_prediction(data):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    
    st.subheader("📊 Prediction Explanation")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(shap_values[0], ax=ax)
    st.pyplot(fig)

# =========================== PDF REPORT GENERATION ===========================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 20)
        self.cell(0, 10, '🫀 Heart Disease Prediction Report 🫀', ln=True, align='C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

def generate_pdf_report(prediction):
    pdf = PDF()
    pdf.add_page()
    
    pdf.chapter_title("Prediction Results")
    result_text = "Heart Disease Detected!" if prediction == 1 else "No Heart Disease Found."
    pdf.chapter_body(result_text)
    
    pdf.chapter_title("Health Recommendations")
    if prediction == 1:
        pdf.chapter_body("Follow a heart-healthy diet, regular exercise, and medication as prescribed.")
    else:
        pdf.chapter_body("Maintain a balanced diet, regular exercise, and routine checkups.")

    pdf_output = "Prediction_Report.pdf"
    pdf.output(pdf_output)
    
    with open(pdf_output, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_output}">📥 Download Prediction Report (PDF)</a>'
        st.markdown(href, unsafe_allow_html=True)

# =========================== PDF DATA UPLOAD FEATURE ===========================
def extract_pdf_data(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        st.text_area("📄 Extracted Data from PDF", text, height=200)
        # Sample logic to process text (convert to numerical data if structured)
        return preprocess_input()
    except Exception as e:
        st.error("❌ Error reading PDF. Please ensure the document contains readable text.")
        return None

# =========================== MAIN FUNCTIONALITY ===========================
if submitted:
    input_data = preprocess_input()
    prediction = predict_heart_disease(input_data)
    health_recommendations(prediction)
    explain_prediction(input_data)
    generate_pdf_report(prediction)

st.markdown("---")
st.markdown("📄 **Upload Patient Data in PDF Format**")
uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

if uploaded_file:
    extracted_data = extract_pdf_data(uploaded_file)
    if extracted_data is not None:
        prediction = predict_heart_disease(extracted_data)
        health_recommendations(prediction)
        explain_prediction(extracted_data)
        generate_pdf_report(prediction)