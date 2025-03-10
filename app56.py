import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont

# --- Generate Dynamic Banner Image ---
def generate_banner():
    img = Image.new("RGB", (900, 200), "#d62828")
    draw = ImageDraw.Draw(img)
    draw.text((50, 70), "🫀 Heart Disease Prediction System", fill="white", 
              font=None, align="center")
    img.save("dynamic_banner.png")

generate_banner()

# Fix model path for Streamlit Cloud
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_heart_disease.pkl")

# Load the trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Set page config
st.set_page_config(
    page_title="🫀 Heart Disease Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling for a Sleek UI
st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; }
        h1 { color: #d62828; text-align: center; font-weight: bold; }
        .stButton>button { background-color: #d62828; color: white; font-size: 18px; border-radius: 10px; }
        .stAlert { font-size: 18px; }
        .suggestion-box { background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107; }
        .success-box { background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745; }
        .error-box { background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 5px solid #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# Display Heart Health Banner
st.image("dynamic_banner.png", use_container_width=True)

# Title
st.markdown("<h1>🫀 Heart Disease Prediction</h1>", unsafe_allow_html=True)

# Input fields
st.sidebar.header("🔹 Enter Patient Details")
age = st.sidebar.number_input("Age", 20, 100, 50)
gender = st.sidebar.selectbox("Gender", [("Male", 1), ("Female", 0)])[1]
height = st.sidebar.number_input("Height (cm)", 100, 220, 165)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
systolic_bp = st.sidebar.number_input("Systolic Blood Pressure", 90, 200, 120)
diastolic_bp = st.sidebar.number_input("Diastolic Blood Pressure", 60, 130, 80)
cholesterol = st.sidebar.selectbox("Cholesterol Level", [("Normal", 1), ("Above Normal", 2), ("High", 3)])[1]
glucose = st.sidebar.selectbox("Glucose Level", [("Normal", 1), ("Above Normal", 2), ("High", 3)])[1]
smoker = st.sidebar.selectbox("Smoker?", [("No", 0), ("Yes", 1)])[1]
alcohol = st.sidebar.selectbox("Alcohol Intake?", [("No", 0), ("Yes", 1)])[1]
physical_activity = st.sidebar.selectbox("Physical Activity?", [("No", 0), ("Yes", 1)])[1]

# Convert input to NumPy array
user_input = np.array([[age, gender, height, weight, systolic_bp, diastolic_bp, cholesterol, glucose, smoker, alcohol, physical_activity]])

# Prediction Logic
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(user_input)[0]

    # Display Result
    if prediction == 1:
        st.error("🚨 **High Risk of Heart Disease!**")
        st.markdown("""<div class='error-box'>
            <h4>🏥 Suggested Actions:</h4>
            - 🩺 **Consult a cardiologist immediately.**<br>
            - 🥗 **Adopt a heart-healthy diet** (less salt, sugar & saturated fats).<br>
            - 🚶 **Increase physical activity** (e.g., 30 mins of walking daily).<br>
            - 💊 **Monitor and manage cholesterol & blood pressure.**<br>
            - 🚭 **Quit smoking & limit alcohol intake.**<br>
            - 😴 **Ensure proper rest & stress management.**
        </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ **No Heart Disease Detected!**")
        st.markdown("""<div class='success-box'>
            <h4>💪 Keep Your Heart Healthy:</h4>
            - 🥦 **Maintain a balanced diet** with plenty of fruits and vegetables.<br>
            - 🏃 **Exercise regularly** for at least 150 minutes per week.<br>
            - ❤️ **Get regular check-ups** to monitor heart health.<br>
            - 🚭 **Avoid smoking & reduce alcohol intake.**<br>
            - 😃 **Practice stress-relief techniques like meditation.**
        </div>""", unsafe_allow_html=True)

    # SHAP Explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(user_input)

    st.subheader("🔍 Feature Importance Analysis")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0])
    st.pyplot(fig)

    # Feature Impact Chart
    feature_names = ["Age", "Gender", "Height", "Weight", "Systolic BP", "Diastolic BP", "Cholesterol", "Glucose", "Smoker", "Alcohol", "Physical Activity"]
    shap_values_array = np.abs(shap_values.values[0])
    
    feature_df = pd.DataFrame({"Feature": feature_names, "Impact": shap_values_array})
    feature_df = feature_df.sort_values("Impact", ascending=False)

    fig = px.bar(
        feature_df, x="Impact", y="Feature", 
        orientation="h", title="📊 Feature Impact on Prediction", 
        color="Impact", color_continuous_scale="reds"
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("<br><hr><center>👨‍⚕️ AI-Powered Heart Health Prediction | Built with ❤️ by Ankush</center>", unsafe_allow_html=True)