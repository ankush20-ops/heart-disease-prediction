import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
from fpdf import FPDF
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# ------------------------ PATH CONFIGURATION ------------------------
PROJECT_PATH = "./"
MODEL_PATH = f"{PROJECT_PATH}xgboost_heart_disease.pkl"

# ------------------------ LOAD MODEL ------------------------
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ------------------------ DYNAMIC BANNER IMAGE ------------------------
# ------------------------ DYNAMIC BANNER IMAGE ------------------------
def create_heart_banner():
    plt.figure(figsize=(10, 2))
    sns.heatmap(np.random.rand(10, 20), cmap='Reds', cbar=False)
    plt.title("❤️ Heart Disease Prediction System", fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.savefig('heart_banner.png', bbox_inches='tight', dpi=300, format='png')
    plt.close()

create_heart_banner()

# ------------------------ STREAMLIT UI ------------------------
def main():
    st.set_page_config(page_title="Heart Disease Prediction System", layout="wide")

    # Display the dynamically generated banner
    try:
        st.image("heart_banner.png", use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Banner image could not be loaded: {e}")
        st.markdown("### ❤️ Heart Disease Prediction System")

    st.title("💓 AI-Powered Heart Disease Prediction System")
    st.markdown("---")

    # Rest of the code remains the same

# ------------------------ EXPECTED FEATURE ORDER ------------------------
expected_features = [
    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", 
    "Exercise Angina", "ST Depression", "Slope", "Major Vessels", "Thal"
]

# ------------------------ DATA CLEANING ------------------------
def clean_and_format_input(data):
    try:
        df = pd.DataFrame([data], columns=expected_features)
        df.fillna(0, inplace=True)
        df = df.astype(float)
        return df.values
    except Exception as e:
        st.error(f"❗ Data Formatting Error: {e}")
        return None

# ------------------------ PREDICTION FUNCTION ------------------------
def predict_heart_disease(data):
    cleaned_data = clean_and_format_input(data)
    if cleaned_data is None:
        st.error("❗ Invalid data format. Please check the input values.")
        return "Error"

    if len(cleaned_data.shape) == 1:
        cleaned_data = cleaned_data.reshape(1, -1)

    try:
        prediction = model.predict(cleaned_data)[0]
        return prediction
    except Exception as e:
        st.error(f"❗ Prediction Error: {e}")
        return "Error"

# ------------------------ DISEASE TYPE IDENTIFICATION ------------------------
def identify_disease(prediction):
    disease_types = {
        1: "Angina (Chest Pain)",
        2: "Myocardial Infarction (Heart Attack)",
        3: "Ischemic Heart Disease",
        4: "Hypertensive Heart Disease"
    }
    return disease_types.get(prediction, "Unknown Heart Condition")

# ------------------------ SUGGESTIONS FUNCTION ------------------------
def get_suggestions(result):
    if result == 0:
        return """
        ✅ **Congratulations! No signs of heart disease were detected.**
        **Prevention Tips:**  
        - Maintain a healthy diet (rich in fruits, veggies, and whole grains).  
        - Exercise regularly (30 minutes/day).  
        - Avoid smoking and limit alcohol.  
        - Manage stress and prioritize sleep.
        """
    else:
        return f"""
        ⚠️ **Heart disease detected: {identify_disease(result)}**  
        **Health Tips:**  
        - Follow a heart-healthy diet (low sodium & cholesterol).  
        - Engage in daily moderate exercises.  
        - Monitor your blood pressure and cholesterol regularly.  
        - Follow your doctor's prescribed medication routine.  
        """

# ------------------------ PDF REPORT GENERATION ------------------------
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 20)
        self.cell(0, 10, 'Heart Disease Prediction Report', ln=True, align='C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, ln=True)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)

def generate_pdf_report(result, data, suggestions):
    pdf = PDFReport()
    pdf.add_page()

    # Patient Details
    pdf.chapter_title("Patient Details:")
    patient_info = "\n".join([f"{feature}: {value}" for feature, value in zip(expected_features, data)])
    pdf.chapter_body(patient_info)

    # Prediction Results
    pdf.chapter_title("Prediction Results:")
    disease_info = "No heart disease detected" if result == 0 else f"Disease Detected: {identify_disease(result)}"
    pdf.chapter_body(disease_info)

    # Suggestions
    pdf.chapter_title("Health Suggestions:")
    pdf.chapter_body(suggestions)

    pdf_path = "Heart_Disease_Report.pdf"
    pdf.output(pdf_path)
    return pdf_path

# ------------------------ SHAP EXPLANATION ------------------------
def explain_prediction(data):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0])
    st.pyplot(plt)

# ------------------------ STREAMLIT UI ------------------------
def main():
    st.set_page_config(page_title="Heart Disease Prediction System", layout="wide")
    st.image("heart_banner.png", use_column_width=True)

    st.title("💓 AI-Powered Heart Disease Prediction System")
    st.markdown("---")

    # 📄 PDF Upload Section
    uploaded_file = st.file_uploader("📄 Upload Patient Data (PDF Format)", type=["pdf"])
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        page_text = pdf_reader.pages[0].extract_text()
        st.text_area("Extracted Data", page_text)
        
        # Assuming extracted data is parsed correctly
        input_data = list(map(float, page_text.strip().split(",")))
    else:
        # Manual Input Form
        input_data = [st.number_input(feature, 0.0) for feature in expected_features]

    if st.button("🔍 Predict"):
        prediction = predict_heart_disease(input_data)
        if prediction == "Error":
            st.error("Prediction Failed. Please recheck the data.")
        else:
            st.success(f"✅ Prediction: {identify_disease(prediction)}")
            st.markdown(get_suggestions(prediction))

            # SHAP Explanation
            st.markdown("### 🔎 Prediction Explanation")
            explain_prediction(np.array([input_data]))

            # Generate Report
            report_path = generate_pdf_report(prediction, input_data, get_suggestions(prediction))
            with open(report_path, "rb") as file:
                st.download_button("📥 Download Detailed Report", file, file_name="Heart_Disease_Report.pdf")

if __name__ == "__main__":
    main()