import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('fistula_closure_model.pkl')

st.title("Fistula Closure Predictor")
st.write("Enter new patient details to predict fistula closure outcome")
st.write("Prediction is for DLPS/DLPF/LIFT with FiLaC")

# Form inputs
grade = st.selectbox("Grade of Fistula (St. James classification of perianal fistulae)", [1, 2, 3, 4, 5])
branched = st.selectbox("Branched Fistula? (0 = No, 1 = Yes)", [0, 1])
inflam = st.selectbox("Inflammation Around the Tract Present? (0 = No, 1 = Yes)", [0, 1])
infection = st.selectbox("Wound Infection Present? (0 = No, 1 = Yes)", [0, 1])
length_fistula = st.slider("Length of Fistula (in mm)", min_value=10, max_value=100, value=50)

# On button click
if st.button("Predict Closure"):
    new_data = pd.DataFrame([{
        'grade': grade,
        'length_fistula': length_fistula,
        'branched': branched,
        'inflam': inflam,
        'infection': infection,
    }])
    
    pred = model.predict(new_data)[0]
    prob = model.predict_proba(new_data)[0][1]
    
    st.subheader("Result:")
    st.success(f"Predicted Closure: {'Yes' if pred == 1 else 'No'}")
    st.info(f"Probability of Closure: {prob:.2%}")


st.write("This predictor model is created from a study in a single hospital. So prediction may vary in your setup")
st.write("© 2025 Dr. Surendra Shah. All rights reserved.")
