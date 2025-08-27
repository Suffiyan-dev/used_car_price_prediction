# 🚗 Used Car Price Prediction App

A machine learning web application built with **Streamlit** that predicts the price of a used car based on user inputs like car age, odometer reading, manufacturer, fuel type, and more. The model is trained on preprocessed and scaled data to ensure better performance and accuracy.

---

## 📂 Project Structure

used_car_price_prediction/
│
├── models/
│ ├── linear_regression_model.pkl
│ ├── feature_scaler.pkl
│ ├── target_scaler.pkl
│ └── model_columns.pkl
├── App.py # Main Streamlit app
├── ead_feature_engineering.ipynb # Model training and saving pipeline
├── requirements.txt # Required Python packages
└── README.md # Project documentation


---

## 🚀 Features

- Predicts used car prices based on user input
- Uses a trained Linear Regression model
- Handles categorical and numerical preprocessing (with `get_dummies` + `MinMaxScaler`)
- Properly reverses normalization for accurate price output
- Fully interactive Streamlit web interface

---

## 📦 Requirements

Install all dependencies using pip:

```bash
pip install -r requirements.txt


🧠 How the Model Works

Categorical variables: One-hot encoded using pd.get_dummies

Numerical variables: Scaled using MinMaxScaler

Target (price): Capped to remove extreme values, then normalized

Model: Trained on normalized features and target

Post-processing: After prediction, price is inverse-transformed back to original scale


▶️ Running the App

Make sure your models and scalers are saved in the models/ folder.

Then run the Streamlit app with:

streamlit run App.py


The app will open in your browser, allowing you to input car details and get a price predictions



🛠️ Training Your Own Model

Use training_script.py to preprocess your dataset, train the model, and save:

linear_regression_model.pkl

feature_scaler.pkl

target_scaler.pkl

model_columns.pkl


📈 Example Inputs

Manufacturer: Toyota

Condition: good

Odometer: 75000

Transmission: automatic

Fuel: gas

Car Age: 6 years



🧑‍💻 Author
Muhammad Suffiyan