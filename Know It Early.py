import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and feature names
classifier, scaler, feature_names = joblib.load("breast_cancer_model.pkl")

# Page config
st.set_page_config(page_title="Know It Early", layout="centered", page_icon="🩺")
st.image("Logo.png", width=120)
st.title("🧬 Know It Early")
st.markdown("""
#### A Clinical Tool for Early Breast Cancer Detection  
Use the form below to input clinical tumor characteristics. This tool will assist in predicting whether the tumor is **malignant** or **benign** based on a trained logistic regression model.
""")

# Sidebar inputs
st.sidebar.header("Enter Tumor Measurements")

def get_user_input():
    user_data = {}
    for feature in feature_names:
        user_data[feature] = st.sidebar.number_input(
            label=feature,
            value=0.0,
            format="%.4f"
        )
    return pd.DataFrame([user_data])

# Collect input
input_df = get_user_input()

# Display input
st.subheader("🧾 Patient Input Data")
st.write(input_df)

# Predict button
if st.button("🔍 Predict Diagnosis"):
    try:
        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = classifier.predict(input_scaled)[0]
        proba = classifier.predict_proba(input_scaled)[0][prediction]

        result = "🛑 Malignant Tumor" if prediction == 1 else "✅ Benign Tumor"

        # Output
        st.subheader("🔬 Prediction Result")
        st.success(f"{result}\n\nConfidence: **{proba * 100:.2f}%**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ for early diagnosis | © 2025 Know It Early")
