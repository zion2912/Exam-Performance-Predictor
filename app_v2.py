import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

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

    # --- Feature Importance Chart ---
    st.subheader("Feature Importance")
    feature_names = ["age", "gender", "study_time_hours", "absences", "parent_education", "previous_score"]
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, palette="viridis")
    st.pyplot(fig)

    # --- Distribution of Scores (Optional Demo Data) ---
    st.subheader("Distribution of Exam Scores Across all Students")
    # Load your dataset for visualization (replace with actual path if needed)
    df = pd.read_csv("student_performance.csv")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["final_exam_score"], bins=20, kde=True, ax=ax2, color="skyblue")
    ax2.set_title("Final Exam Score Distribution")
    st.pyplot(fig2)
