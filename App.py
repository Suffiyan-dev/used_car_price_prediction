import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Used Car Price Prediction App")

model = joblib.load(r"D:\used_car_price_prediction\models\linear_regression_model.pkl")
scaler_features = joblib.load(r"D:\used_car_price_prediction\models\scaler_features.pkl")
model_columns = joblib.load(r"D:\used_car_price_prediction\models\model_columns.pkl")
scaler_target = joblib.load(r"D:\used_car_price_prediction\models\scaler_target.pkl")

manufacturer = st.selectbox("Manufacturer", ['gmc', 'chevrolet', 'toyota', 'ford', 'jeep', 'nissan', 'ram',
       'mazda', 'cadillac', 'honda', 'dodge', 'lexus', 'jaguar', 'buick',
       'chrysler', 'volvo', 'audi', 'infiniti', 'lincoln', 'alfa-romeo',
       'subaru', 'acura', 'hyundai', 'mercedes-benz', 'bmw', 'mitsubishi',
       'volkswagen', 'porsche', 'kia', 'rover', 'ferrari', 'mini',
       'pontiac', 'fiat', 'tesla', 'saturn', 'mercury', 'harley-davidson',
       'datsun', 'aston-martin', 'land rover', 'morgan'])  
condition = st.selectbox("Condition", ['good','fair', 'like new', 'new', 'salvage','unknown'])
cylinders = st.selectbox("Cylinders", ['8 cylinders', '6 cylinders', '4 cylinders', '5 cylinders',
       'other', '3 cylinders','12 cylinders'])
fuel = st.selectbox("Fuel Type", ['gas', 'other','hybrid', 'electric'])
odometer = st.number_input("Odometer Reading", min_value=0)
title_status = st.selectbox("Title Status", ['unknown', 'rebuilt', 'lien', 'salvage', 'missing',
       'parts only'])
transmission = st.selectbox("Transmission", ['manual', 'other','unknown'])
drive = st.selectbox("Drive Type", ['fwd', 'rwd','unknown'])
size = st.selectbox("Car Size", ['sub-compact', 'mid-size', 'full-size','unknown'])
car_type = st.selectbox("Car Type", ['pickup', 'truck', 'other', 'coupe','hatchback',
       'mini-van', 'sedan', 'offroad', 'bus', 'van', 'convertible',
       'wagon','unknown'])
car_age = st.number_input("Car Age", min_value=0)
year = st.number_input("Year of Manufacture", min_value=1900, max_value=2025)

user_input = pd.DataFrame([{
    'manufacturer': manufacturer,
    'condition': condition,
    'cylinders': cylinders,
    'fuel': fuel,
    'odometer': odometer,
    'title_status': title_status,
    'transmission': transmission,
    'drive': drive,
    'size': size,
    'type': car_type,
    'car_age': car_age,
    'year': year
}])

categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 'title_status',
                    'transmission', 'drive', 'size', 'type']
numerical_cols = ['odometer', 'car_age']
user_encoded = pd.get_dummies(user_input, columns=categorical_cols, drop_first=True)
user_encoded = user_encoded.reindex(columns=model_columns, fill_value=0)
user_encoded[numerical_cols] = scaler_features.transform(user_encoded[numerical_cols])

if st.button("Predict"):
    pred_norm = model.predict(user_encoded) 
    pred_actual = scaler_target.inverse_transform(pred_norm.reshape(-1, 1))  
    st.success(f"Estimated Car Price: ${pred_actual[0][0]:,.2f}")
