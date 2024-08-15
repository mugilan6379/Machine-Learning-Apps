import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



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
    #form.form_submit_button("Submit")
    diabetes_dataframe=pd.read_csv('Data/diabetes.csv')
    Y=diabetes_dataframe['Outcome']
    X=diabetes_dataframe.drop(columns='Outcome',axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data=scaler.transform(X)
    X=standardized_data
    Y=diabetes_dataframe['Outcome']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
    classifier=svm.SVC(kernel='linear')
    classifier.fit(X_train,Y_train)
    X_train_prediction = classifier.predict(X_train)
    train_accuracy=accuracy_score(X_train_prediction,Y_train)
    X_test_prediction = classifier.predict(X_test)
    test_accuracy=accuracy_score(X_test_prediction,Y_test)
    #Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age
    input_data = (preg,Glucose,bp,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,age)

    nparray = np.asarray(input_data)

    data_reshaped=nparray.reshape(1,-1)


    standard_data = scaler.transform(data_reshaped)
    print(standard_data)

    predicted_data = classifier.predict(standard_data)
    print(predicted_data)

    if (predicted_data[0]==0):
        st.write("The person is not diabetic")
    else:
        st.write('The person is diabetic')
    st.write('Training Accuracy Score: ',train_accuracy)    
    st.write('Accuracy Score is: ',test_accuracy)
    
