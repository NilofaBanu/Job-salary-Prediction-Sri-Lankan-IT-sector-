import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("salary_model.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="IT Salary Predictor", layout="centered")
#window key + .
st.title("💻 IT Job Salary Prediction System")
st.write("Enter job details to predict expected salary (LKR)")

# Input fields
role = st.selectbox("Select Job Role",
                    ["Software Engineer", "Data Scientist", "Web Developer",
                     "System Administrator", "Network Engineer", "QA Engineer"])

soft_skills = st.text_input("Soft Skills (comma separated)",
                            "Communication, Teamwork")

tech_stack = st.text_input("Tech Stack (comma separated)",
                           "Python, SQL")

experience = st.selectbox("Experience (Years)",
                           ["0-1", "1-3", "3-5", "5-10", "10+"])


if st.button("Predict Salary"):

    new_data = pd.DataFrame({
        "Role": [role],
        "Soft Skills": [soft_skills],
        "Tech Stack": [tech_stack],
        "Experience (Years)": [experience]
    })

    new_data_enc = encoder.transform(new_data)
    prediction = model.predict(new_data_enc)

    st.success(f"Predicted Salary: LKR {round(prediction[0], 2)}")

