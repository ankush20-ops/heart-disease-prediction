import streamlit as st
import numpy as np
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# ✅ Set correct file paths
MODEL_PATH = r"C:\Users\jaiba\Downloads\AI_Healthcare_Project\xgboost_heart_disease.pkl"
IMAGE_PATH = r"C:\Users\jaiba\OneDrive\Documents\heart_banner.jpg.webp"

# ✅ Load the trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ✅ UI Enhancements
st.markdown("<h1 style='text-align: center;'>🫀 Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.image(IMAGE_PATH, use_container_width=True)

# ✅ Custom Styling
st.markdown("""
    <style>
        .stApp { background-color: #f5f5f5; }
        h1 { color: #d62828; text-align: center; }
        .css-1cpxqw2 { font-size: 18px !important; }
    </style>
    """, unsafe_allow_html=True)

# ✅ User Input Fields
st.sidebar.header("Enter Your Health Details")
age = st.sidebar.slider("Age", 20, 90, 50)
gender = st.sidebar.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
height = st.sidebar.number_input("Height (cm)", 120, 220, 165)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
ap_hi = st.sidebar.number_input("Systolic Blood Pressure", 90, 200, 120)
ap_lo = st.sidebar.number_input("Diastolic Blood Pressure", 60, 150, 80)
cholesterol = st.sidebar.selectbox("Cholesterol Level", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
glucose = st.sidebar.selectbox("Glucose Level", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
smoke = st.sidebar.radio("Smoker?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
alco = st.sidebar.radio("Alcohol Intake?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
active = st.sidebar.radio("Physical Activity?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# ✅ Prepare Input Data for Model
input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, alco, active]])

# ✅ Prediction
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk: You might have heart disease. Please consult a doctor.")
    else:
        st.success("✅ Low Risk: You are less likely to have heart disease.")

    # ✅ SHAP Explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    st.subheader("📊 Feature Importance Analysis")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0])
    st.pyplot(fig)

    # ✅ Health Recommendations
    st.subheader("💡 Health Recommendations")
    if prediction == 1:
        st.write("""
        - 🏥 **Consult a doctor immediately** for further tests.
        - 🥗 **Eat a heart-healthy diet** (low in saturated fats & high in fiber).
        - 🚶 **Increase physical activity** (30 mins daily walking is beneficial).
        - 🚭 **Quit smoking & reduce alcohol intake**.
        - 💊 **Monitor blood pressure & cholesterol levels**.
        """)
    else:
        st.write("""
        - ✅ Maintain a **healthy lifestyle** to prevent heart disease.
        - 🏃 **Exercise regularly** to keep your heart strong.
        - 🍎 **Eat balanced meals** with fruits & vegetables.
        - 🩺 **Get regular checkups** to monitor heart health.
        - 💖 **Avoid stress** and practice meditation or yoga.
        """)

# ✅ Data Visualization
st.subheader("📈 Heart Disease Statistics")
data = pd.DataFrame({
    "Category": ["With Disease", "Without Disease"],
    "Count": [500, 1000]  # Example numbers
})
fig = px.pie(data, names="Category", values="Count", title="Heart Disease Distribution")
st.plotly_chart(fig)
