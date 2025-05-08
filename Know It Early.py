import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your trained model
with open('breast_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="Know It Early", layout="centered", page_icon="🩺")

# Optional: Add logo
st.image("Logo.png", width=120)  # Ensure logo.png is in the same folder
st.title("🧬 Know It Early")
st.markdown("""
#### A Clinical Tool for Early Breast Cancer Detection  
Use the form below to input clinical tumor characteristics. This tool will assist in predicting whether the tumor is **malignant** or **benign** based on a trained logistic regression model.
""")

# Sidebar input form
st.sidebar.header("Enter Tumor Measurements")

def get_user_input():
    mean_radius = st.sidebar.slider("Mean Radius", 6.0, 30.0, 14.0)
    mean_texture = st.sidebar.slider("Mean Texture", 9.0, 40.0, 20.0)
    mean_perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, 90.0)
    mean_area = st.sidebar.slider("Mean Area", 140.0, 2500.0, 500.0)
    mean_smoothness = st.sidebar.slider("Mean Smoothness", 0.05, 0.2, 0.1)

    data = {
        'mean radius': mean_radius,
        'mean texture': mean_texture,
        'mean perimeter': mean_perimeter,
        'mean area': mean_area,
        'mean smoothness': mean_smoothness
    }

    return pd.DataFrame([data])

# Collect input
input_df = get_user_input()

# Display input
st.subheader("🧾 Patient Input Data")
st.write(input_df)

# Prediction
if st.button("🔍 Predict Diagnosis"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction]
    result = "🛑 Malignant Tumor" if prediction == 0 else "✅ Benign Tumor"

    st.subheader("🔬 Prediction Result")
    st.success(f"{result}\n\nConfidence: **{proba * 100:.2f}%**")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ for early diagnosis | © 2025 Know It Early")
