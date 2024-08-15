import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline  import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,r2_score
import matplotlib.pyplot as plt
import warnings

def pred():
    if 'page' in st.session_state and st.session_state.page == "carPrice_prediction":
        st.write("# Car Prediction Model Details")
        st.write("This page provides a detailed explanation of the model, including the features used and how predictions are made.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.experimental_rerun()
    #Year	Present_Price	Kms_Driven	Fuel_Type	Seller_Type	Transmission	Owner
    form=st.form('car_form')
    year = form.number_input(label='Year')
    present_price = form.number_input(label='Present price')
    kms_driven = form.number_input(label='Kilometres Driven')
    fuel_type = form.selectbox('Fuel type',['Petrol','Diesel','CNG'])
    seller_type = form.selectbox('Seller type',['Dealer','Individual'])
    transmission = form.selectbox('Transmission',['Manual','Automatic'])
    owner = form.number_input(label='Owner')
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    car_details = pd.read_csv('Data/carData.csv')
    X = car_details.drop(columns=['Car_Name','Selling_Price'])
    Y = car_details['Selling_Price']
    cat_cols = X.select_dtypes(include='object').columns
    num_cols = X.select_dtypes(exclude='object').columns
    scalar = StandardScaler()
    encoder = OneHotEncoder()
    transformer = ColumnTransformer([
        ('StandardScaler',scalar,num_cols),
        ('OneHotEncoder',encoder,cat_cols)
    ])


    pipeline = Pipeline(steps=[
        ('preprocessor', transformer),
        ('model', LinearRegression())
    ])

    param_grid = [
        {'model':[LinearRegression()]},
        {'model':[Lasso()],'model__alpha': [0.1,1.0,10.0]}
    ]

    grid_search = GridSearchCV(pipeline,param_grid,scoring='r2')
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

    grid_search.fit(X_train,Y_train)
    best_model = grid_search.best_estimator_
    Y_train_pred=best_model.predict(X_train)
    Y_test_pred=best_model.predict(X_test)
    print('Best model is',grid_search.best_params_['model'])
    print('Training r2 score: ',r2_score(Y_train,Y_train_pred))
    print('Test Set r2 score: ',r2_score(Y_test,Y_test_pred))
    plt.scatter(Y_train, Y_train_pred)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title('Actual Price VS Predicted price')
    plt.show()
    
    import numpy as np
    input_data = (year,present_price,kms_driven,fuel_type,seller_type,transmission,owner)
    nparray = np.asarray(input_data)

    data_reshaped=nparray.reshape(1,-1)
    columns = ['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']

    input_df = pd.DataFrame(data_reshaped, columns=columns)
    prediction = best_model.predict(input_df)
    st.write(f'Predicted value: {prediction[0]}')