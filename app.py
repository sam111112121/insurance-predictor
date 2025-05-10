import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Insurance Predictor", layout="centered")

# Logo with colored background
st.markdown("""
<div style='background-color: #e3f2fd; padding: 10px 0; width: 100%; text-align: center;'>
    <img src='https://raw.githubusercontent.com/sam111112121/insurance-predictor/main/Sarooj-Sazeh-Tabnak.png' width='200'/>
</div>
""", unsafe_allow_html=True)

# Welcome message
if 'language' not in st.session_state:
    st.session_state["language"] = "English"

language = st.selectbox("🌐 Choose Language / Sprache wählen:", ["English", "Deutsch"], index=0 if st.session_state["language"] == "English" else 1)
st.session_state["language"] = language

st.markdown("""
<div style='background-color: #f0f4ff; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
    <h3 style='margin: 0;'>👋 {}<h3>
    <p style='margin: 0;'>{}</p>
</div>
""".format(
    "Welcome to the Insurance Cost Predictor App!" if language == "English" else
    "Willkommen zur Versicherungskosten-Vorhersage-App!",
    "Fill out the form to estimate your annual insurance cost using our trained model." if language == "English" else
    "Füllen Sie das Formular aus, um Ihre jährlichen Versicherungskosten mithilfe unseres Modells zu schätzen."
), unsafe_allow_html=True)

# Styling
st.markdown("""
<style>
    .main {background-color: #f4f6f8;}
    h1, h2, h3 {color: #003566;}
    .stButton button {
        background-color: #0077b6;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
</style>
""", unsafe_allow_html=True)

# Session state for history
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=[
        'Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region', 'Model', 'Predicted Cost (€)'
    ])

# Title
st.title("💰 Insurance Cost Prediction App" if language == "English" else "💰 Versicherungskosten-Vorhersage")

# Inputs
age = st.slider("Age" if language == "English" else "Alter", 18, 64, 30)
sex = st.selectbox("Sex" if language == "English" else "Geschlecht", ['male', 'female'] if language == "English" else ['männlich', 'weiblich'])

# دریافت قد و وزن از کاربر
height = st.number_input("Enter your height (in cm)" if language == "English" else "Geben Sie Ihre Höhe (in cm) ein", min_value=50, max_value=250, value=170)
weight = st.number_input("Enter your weight (in kg)" if language == "English" else "Geben Sie Ihr Gewicht (in kg) ein", min_value=30, max_value=200, value=70)

# محاسبه BMI
bmi = weight / ((height / 100) ** 2)

children = st.number_input("Number of Children" if language == "English" else "Anzahl der Kinder", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker" if language == "English" else "Raucher", ['yes', 'no'] if language == "English" else ['ja', 'nein'])
region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])
model_choice = st.selectbox("Choose prediction model:" if language == "English" else "Modell auswählen:", ["Random Forest", "Linear Regression"])

# Encode inputs
sex_val = 0 if (sex == 'male' or sex == 'männlich') else 1
smoker_val = 1 if (smoker == 'yes' or smoker == 'ja') else 0
region_northwest = 1 if region == 'northwest' else 0
region_southeast = 1 if region == 'southeast' else 0
region_southwest = 1 if region == 'southwest' else 0

input_data = np.array([[age, sex_val, bmi, children, smoker_val,
                        region_northwest, region_southeast, region_southwest]])

# Prediction
if st.button("Predict Insurance Cost" if language == "English" else "Versicherungskosten vorhersagen"):
    if model_choice == "Linear Regression":
        model = joblib.load("insurance_linear_model.pkl")
    else:
        model = joblib.load("insurance_rf_model.pkl")

    prediction = model.predict(input_data)[0]  # مقدار پیش‌بینی را در متغیر prediction ذخیره می‌کنیم

    msg = f"💡 Estimated Annual Insurance Cost: €{prediction:,.2f}" if language == "English" \
        else f"💡 Geschätzte jährliche Versicherungskosten: €{prediction:,.2f}"
    st.success(msg)

    new_entry = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'Children': children,
        'Smoker': smoker,
        'Region': region,
        'Model': model_choice,
        'Predicted Cost (€)': round(prediction, 2)
    }])
    st.session_state['history'] = pd.concat([st.session_state['history'], new_entry], ignore_index=True)

# Display history
if not st.session_state['history'].empty:
    st.markdown("## 📊 Prediction History Table" if language == "English" else "## 📊 Tabelle der Vorhersagehistorie")
    st.dataframe(st.session_state['history'])

    # Download CSV
    csv = st.session_state['history'].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV" if language == "English" else "⬇️ CSV herunterladen", csv, "prediction_history.csv", "text/csv")
