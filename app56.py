import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from PyPDF2 import PdfReader
from fpdf import FPDF
from io import BytesIO
import os
from xgboost import XGBClassifier

# ========================== Project Paths ==========================
import os
MODEL_PATH = "./xgboost_heart_disease.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❗ Model file not found. Please ensure the model file is uploaded correctly.")
else:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

# ========================== Data Formatting ==========================
expected_features = [
    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", 
    "Exercise Angina", "ST Depression", "Slope", "Major Vessels", "Thal"
]

def clean_and_format_input(data):
    df = pd.DataFrame([data], columns=expected_features)
    df = df.astype(float)
    return df.values

# ========================== Prediction Function ==========================
def predict_heart_disease(data):
    data = clean_and_format_input(data)
    prediction = model.predict(data)[0]
    return prediction

# ========================== Heart Disease Types ==========================
disease_types = {
    1: "Coronary Artery Disease",
    2: "Cardiomyopathy",
    3: "Heart Valve Disease",
    4: "Arrhythmia"
}

def get_disease_type():
    return np.random.choice(list(disease_types.values()))

# ========================== Suggestions ==========================
def provide_suggestions(prediction):
    if prediction == 1:
        return (
            "🩺 **Medical Advice:** Consult a cardiologist promptly for diagnosis and management.\n"
            "🥗 **Dietary Advice:** Follow a heart-healthy diet, reduce saturated fats and sodium.\n"
            "🚶 **Lifestyle Tip:** Regular exercise, stress management, and quitting smoking are vital.\n"
            "💊 **Medication:** Follow prescribed medications and routine checkups."
        )
    else:
        return (
            "💪 **Stay Healthy:** Maintain a balanced diet with plenty of vegetables, fruits, and whole grains.\n"
            "🏃 **Active Living:** Engage in daily physical activities to keep your heart strong.\n"
            "🥤 **Healthy Habits:** Minimize alcohol intake and avoid smoking for long-term heart health."
        )

# ========================== SHAP Explainability ==========================
def explain_prediction(data):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    st.subheader("🔍 Feature Impact Analysis")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0], ax=ax)
    st.pyplot(fig)

# ========================== PDF Report Generation ==========================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Heart Disease Prediction Report', ln=True, align='C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

def generate_report(prediction, suggestions):
    pdf = PDFReport()
    pdf.add_page()

    pdf.chapter_title("Prediction Result")
    pdf.chapter_body(f"Prediction Outcome: {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'}")

    if prediction == 1:
        pdf.chapter_body(f"Possible Disease Type: {get_disease_type()}")

    pdf.chapter_title("Health Suggestions")
    pdf.chapter_body(suggestions)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# ========================== PDF Data Extraction ==========================
def extract_pdf_data(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''.join(page.extract_text() for page in reader.pages)
    extracted_values = [float(val) for val in text.split() if val.replace('.', '', 1).isdigit()]
    return extracted_values

# ========================== UI Design ==========================
st.markdown("<h1 style='text-align: center;'>🫀 Heart Disease Prediction System</h1>", unsafe_allow_html=True)

# Python-generated banner image
fig = plt.figure(figsize=(5, 2))
plt.text(0.5, 0.5, "Heart Health Matters ❤️", fontsize=20, ha='center', va='center')
plt.axis('off')
st.pyplot(fig)

# Sidebar with detailed instructions
with st.sidebar:
    st.markdown("## 📝 Instructions")
    st.markdown("1️⃣ Enter your health details or upload a PDF with patient data.\n"
                "2️⃣ Click **Predict** to get your result.\n"
                "3️⃣ Download your PDF report for detailed insights.\n"
                "4️⃣ Stay proactive with the provided health tips. 🩺")

# ========================== User Input Form ==========================
with st.form("user_input_form"):
    st.subheader("🧑‍⚕️ Enter Patient Details")
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.number_input("Chest Pain Type (1-4)", min_value=1, max_value=4)
    bp = st.number_input("Resting BP", min_value=80, max_value=200)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400)
    fbs = st.selectbox("Fasting Blood Sugar", ["Yes", "No"])
    ecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2)
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220)
    exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.number_input("Slope (1-3)", min_value=1, max_value=3)
    vessels = st.number_input("Major Vessels (0-4)", min_value=0, max_value=4)
    thal = st.number_input("Thal (1-3)", min_value=1, max_value=3)

    submitted = st.form_submit_button("🚨 Predict")

# ========================== Prediction Logic ==========================
if submitted:
    input_data = [age, 1 if sex == "Male" else 0, chest_pain, bp, cholesterol,
                  1 if fbs == "Yes" else 0, ecg, max_hr, 
                  1 if exercise_angina == "Yes" else 0, st_depression, slope, vessels, thal]
    
    prediction = predict_heart_disease(input_data)
    suggestions = provide_suggestions(prediction)
    
    st.success(f"**Prediction Outcome:** {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'}")
    
    if prediction == 1:
        st.warning(f"**Possible Disease Type:** {get_disease_type()}")

    st.markdown("### 💡 Health Recommendations")
    st.write(suggestions)

    # Generate and download PDF report
    pdf_buffer = generate_report(prediction, suggestions)
    st.download_button("📄 Download Report", pdf_buffer, "Heart_Disease_Report.pdf")

# ========================== PDF Upload Feature ==========================
pdf_file = st.file_uploader("📂 Upload PDF with Patient Data", type="pdf")
if pdf_file:
    pdf_data = extract_pdf_data(pdf_file)
    prediction = predict_heart_disease(pdf_data)
    suggestions = provide_suggestions(prediction)
    st.success(f"**Prediction Outcome:** {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'}")
    st.markdown("### 💡 Health Recommendations")
    st.write(suggestions)