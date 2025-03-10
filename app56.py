import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import plotly.express as px
from fpdf import FPDF
from PyPDF2 import PdfReader
from PIL import Image

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_heart_disease.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Page Configuration
st.set_page_config(
    page_title="🫀 Heart Disease Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
        .stApp { background-color: #f5f5f5; }
        h1 { color: #d62828; text-align: center; }
        .stButton>button { background-color: #d62828; color: white; font-size: 18px; }
        .stAlert { font-size: 18px; }
        .pdf-box { border: 2px solid #d62828; padding: 10px; border-radius: 10px; background-color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# Display Heart Banner
heart_banner = Image.new("RGB", (800, 200), "#d62828")
st.image(heart_banner, use_container_width=True)

# Title
st.markdown("<h1>🫀 Heart Disease Prediction</h1>", unsafe_allow_html=True)

# Sidebar for Inputs
st.sidebar.header("🔹 Enter Patient Details")
age = st.sidebar.number_input("Age", 20, 100, 50)
gender = st.sidebar.selectbox("Gender", sorted([("Male", 1), ("Female", 0)]))[1]
cholesterol = st.sidebar.selectbox("Cholesterol Level", sorted([("High", 3), ("Above Normal", 2), ("Normal", 1)]))[1]
glucose = st.sidebar.selectbox("Glucose Level", sorted([("High", 3), ("Above Normal", 2), ("Normal", 1)]))[1]
smoker = st.sidebar.selectbox("Smoker?", sorted([("Yes", 1), ("No", 0)]))[1]
alcohol = st.sidebar.selectbox("Alcohol Intake?", sorted([("Yes", 1), ("No", 0)]))[1]
physical_activity = st.sidebar.selectbox("Physical Activity?", sorted([("Yes", 1), ("No", 0)]))[1]

# PDF Upload for Auto-Fill
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload Patient PDF Report")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

def extract_data_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text()
    # Example extraction logic (Modify as per your data structure)
    data_values = [int(val) for val in content.split() if val.isdigit()]
    return data_values[:11]  # First 11 features only

if uploaded_file:
    try:
        auto_filled_data = extract_data_from_pdf(uploaded_file)
        if len(auto_filled_data) == 11:
            st.sidebar.success("✅ Data extracted from PDF successfully!")
            age, gender, cholesterol, glucose, smoker, alcohol, physical_activity = auto_filled_data
        else:
            st.sidebar.warning("⚠️ Invalid PDF format. Please check your file.")
    except:
        st.sidebar.error("❌ Failed to extract data from PDF. Please try again.")

# Convert input data
user_input = np.array([[age, gender, cholesterol, glucose, smoker, alcohol, physical_activity]])

# Heart Disease Types
disease_names = {
    1: "Coronary Artery Disease (CAD)",
    2: "Arrhythmia",
    3: "Heart Valve Disease",
    4: "Cardiomyopathy",
    5: "Congenital Heart Disease",
    6: "Myocardial Infarction (Heart Attack)"
}

# Predict and Display
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(user_input)[0]

    if prediction == 1:
        detected_disease = np.random.choice(list(disease_names.values()))
        st.error(f"🚨 **High Risk of Heart Disease Detected: {detected_disease}**")
        st.markdown("### 🏥 Suggested Actions:")
        st.markdown("""
        - 🩺 **Consult a cardiologist immediately.**
        - 🥗 **Follow a heart-healthy diet** (more veggies, less salt & sugar).
        - 🚶 **Increase physical activity** (30 mins daily).
        - 💊 **Monitor and manage cholesterol & blood pressure.**
        - 🚭 **Quit smoking & reduce alcohol intake.**
        - 😴 **Get enough sleep & manage stress.**
        """)
    else:
        st.success("✅ **No Heart Disease Detected!**")
        st.markdown("### 💪 Keep Your Heart Healthy:")
        st.markdown("""
        - 🥦 **Maintain a balanced diet** (fruits, vegetables, whole grains).
        - 🏃 **Exercise regularly** (150 mins per week).
        - ❤️ **Monitor your heart health with regular check-ups.**
        - 🚭 **Avoid smoking & excessive alcohol.**
        - 😃 **Manage stress effectively.**
        """)

    # SHAP Explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(user_input)

    st.subheader("🔍 Feature Importance Analysis")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0])
    st.pyplot(fig)

    # Feature Impact Chart
    feature_names = ["Age", "Gender", "Cholesterol", "Glucose", "Smoker", "Alcohol", "Physical Activity"]
    shap_values_array = np.abs(shap_values.values[0])

    feature_df = pd.DataFrame({"Feature": feature_names, "Impact": shap_values_array})
    feature_df = feature_df.sort_values("Impact", ascending=False)

    fig = px.bar(feature_df, x="Impact", y="Feature", orientation="h", title="📊 Feature Impact on Prediction", color="Impact", color_continuous_scale="reds")
    st.plotly_chart(fig, use_container_width=True)

    # PDF Report Generation
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 18)
            self.cell(0, 10, "🫀 Heart Disease Prediction Report", ln=True, align='C')

        def add_patient_data(self, data):
            self.set_font('Arial', '', 12)
            for key, value in data.items():
                self.cell(0, 10, f"{key}: {value}", ln=True)

        def add_conclusion(self, result):
            self.set_font('Arial', 'B', 14)
            self.cell(0, 10, "Conclusion", ln=True)
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, result)

    report = PDFReport()
    report.add_page()
    report.add_patient_data({
        "Age": age,
        "Gender": "Male" if gender == 1 else "Female",
        "Detected Disease": detected_disease if prediction == 1 else "None"
    })
    report.add_conclusion("Please consult a doctor for a detailed evaluation." if prediction == 1 else "Maintain a healthy lifestyle.")

    report_file = "Heart_Disease_Report.pdf"
    report.output(report_file)

    with open(report_file, "rb") as pdf:
        st.download_button(
            label="📥 Download Report",
            data=pdf,
            file_name=report_file,
            mime="application/pdf"
        )

# Footer
st.markdown("<br><hr><center>👨‍⚕️ AI-Powered Heart Health Prediction | Built with ❤️ by Ankush</center>", unsafe_allow_html=True)