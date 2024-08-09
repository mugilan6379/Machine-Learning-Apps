import streamlit as st
def dia_pred():
    if 'page' in st.session_state and st.session_state.page == "diabetes_prediction":
        st.write("# Diabetes Prediction Model Details")
        st.write("This page provides a detailed explanation of the model, including the features used and how predictions are made.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.experimental_rerun()
    form = st.form('my_form')
    preg=form.number_input(label='No.of Pregnancies: ')
    Glucose=form.number_input(label='Glucose count: ')
    bp = form.number_input(label='Blood Pressure level: ')
    SkinThickness = form.number_input(label='Enter SkinThickness: ')
    Insulin = form.number_input(label='Insulin level: ')
    BMI = form.number_input(label='BMI: ')
    DiabetesPedigreeFunction = form.number_input(label='DiabetesPedigreeFunction: ')
    age = form.number_input(label='AGE: ')
    form.form_submit_button("Submit")
