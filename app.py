import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
rf_model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Exam Performance Predictor")

st.write("Enter student details below to predict whether they will pass or fail.")

# Input fields
age = st.number_input("Age", min_value=15, max_value=22, value=18)
gender = st.selectbox("Gender", ["Male", "Female"])
study_time = st.slider("Study Time (hours per week)", 1, 20, 10)
absences = st.slider("Absences", 0, 30, 2)
parent_education = st.selectbox("Parent Education", ["None", "High School", "Bachelor", "Master"])
previous_score = st.slider("Previous Score", 0, 100, 75)

# Encode categorical inputs
gender_encoded = 1 if gender == "Male" else 0
parent_map = {"None":0, "High School":1, "Bachelor":2, "Master":3}
parent_encoded = parent_map[parent_education]

# Build dataframe for prediction
student_data = pd.DataFrame([{
    "age": age,
    "gender": gender_encoded,
    "study_time_hours": study_time,
    "absences": absences,
    "parent_education": parent_encoded,
    "previous_score": previous_score
}])

# Scale features
X_new = scaler.transform(student_data)

# Predict
if st.button("Predict"):
    prediction = rf_model.predict(X_new)[0]
    prob = rf_model.predict_proba(X_new)[0][prediction]

    st.subheader("Result:")
    st.write("✅ Pass" if prediction == 1 else "❌ Fail")
    st.write(f"Confidence: {prob:.2f}")
