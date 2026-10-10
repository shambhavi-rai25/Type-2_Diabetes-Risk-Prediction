import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_files():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_files()

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="centered"
)

st.title("Diabetes Risk Prediction")
st.write(
    "Enter patient's details below to predict diabetes risk."
)

st.subheader("Patient Information")

age = st.number_input(
    "Age", min_value=0, max_value=120, value=25
)

gender = st.selectbox(
    "Gender", ["Female", "Male"]
)

hypertension = st.selectbox(
    "Hypertension", ["No", "Yes"]
)

heart_disease = st.selectbox(
    "Heart Disease", ["No", "Yes"]
)

smoking_history = st.selectbox(
    "Smoking History",
    ["No Info", "current", "ever", "former", "never"]
)

st.subheader("Health Measurements")

bmi = st.number_input(
    "BMI", min_value=0.0, max_value=100.0,
    value=25.0, step=0.1
)

hba1c = st.number_input(
    "HbA1c Level", min_value=0.0,
    max_value=20.0, value=5.5, step=0.1
)

blood_glucose = st.number_input(
    "Blood Glucose Level (mg/dL)",
    min_value=0.0, max_value=1000.0,
    value=100.0, step=1.0
)

st.divider()

if st.button("Predict Diabetes Risk", use_container_width=True):

    gender_map = {
        "Female": 0,
        "Male": 1
    }

    smoking_map = {
        "No Info": 0,
        "current": 1,
        "ever": 2,
        "former": 3,
        "never": 4
    }

    input_data = pd.DataFrame([[
        gender_map[gender],
        age,
        int(hypertension == "Yes"),
        int(heart_disease == "Yes"),
        smoking_map[smoking_history],
        bmi,
        hba1c,
        blood_glucose
    ]], columns=[
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ])

    try:
        # scaled_input = scaler.transform(input_data)

        prediction = model.predict(input_data)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error(
                "The model predicts a higher likelihood "
                "of diabetes."
            )
        else:
            st.success(
                "The model predicts a lower likelihood "
                "of diabetes."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.write(
                f"Model-estimated diabetes probability: "
                f"{probabilities[1] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Check that the model and scaler use the same "
            "feature order and compatible scikit-learn versions."
        )
