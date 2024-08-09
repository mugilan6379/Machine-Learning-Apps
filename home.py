import streamlit as st
import pandas as pd 
from streamlit_card import card
import diabetes_prediction as dp

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = "home"


if st.session_state.page == "home":
    with st.container():
        st.subheader("Hi, I am Mugilan :wave:")
        st.title("A Data Analyst From United States!")
        st.write(
            "This website showcases my machine learning projects using Streamlit."
        )

    col1,col2,col3 = st.columns(3)

    with col1:
        hasClicked = card(
        title="Diabetes Prediction",
        text="The model predicts whether a person has diabetes based on information such as blood pressure, glucose levels, age, BMI, and more.",
        image="https://imgs.search.brave.com/YgTO7V0XsGFX4NNDpYxk2lKj34ZJhRh5BDhvDEIV7Nw/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJzLmNvbS9p/bWFnZXMvZmVhdHVy/ZWQvcGxhaW4tamVi/OWdhanZ2Y285b2x2/bi5qcGc",
        #url="https://github.com/mugilan6379/Diabetes_Prediction",
        styles= {
            "card" : {
                'padding' : '5px'
            }
        },
        )
        if hasClicked:
            st.session_state.page = "diabetes_prediction"
            st.experimental_rerun()


elif st.session_state.page == "diabetes_prediction":
    dp.dia_pred()