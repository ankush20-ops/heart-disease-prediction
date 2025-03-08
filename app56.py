import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import plotly.express as px
from PIL import Image

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

# Custom styling
st.markdown("""
    <style>
        .stApp { background-color: #f5f5f5; }
        h1 { color: #d62828; text-align: center; }
        .stButton>button { background-color: #d62828; color: white; font-size: 18px; }
        .stAlert { font-size: 18px; }
    </style>
    """, unsafe_allow_html=True)

# Display heart health banner image
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "heart_banner.jpg")
if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, use_container_width=True)

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

# Predict
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(user_input)[0]  # 0 = No disease, 1 = Disease
    
    # Display result
    if prediction == 1:
        st.error("🚨 **High Risk of Heart Disease!**")
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
    feature_names = ["Age", "Gender", "Height", "Weight", "Systolic BP", "Diastolic BP", "Cholesterol", "Glucose", "Smoker", "Alcohol", "Physical Activity"]
    shap_values_array = np.abs(shap_values.values[0])
    
    feature_df = pd.DataFrame({"Feature": feature_names, "Impact": shap_values_array})
    feature_df = feature_df.sort_values("Impact", ascending=False)

    fig = px.bar(feature_df, x="Impact", y="Feature", orientation="h", title="📊 Feature Impact on Prediction", color="Impact", color_continuous_scale="reds")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("<br><hr><center>👨‍⚕️ AI-Powered Heart Health Prediction | Built with ❤️ by Ankush</center>", unsafe_allow_html=True)