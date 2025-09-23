import streamlit as st
import requests
import time
import os

API_URL = "http://model_api:8000/predict"


MODEL_PATH = "/opt/airflow/final_model/cool_model.pkl"

while not os.path.exists(MODEL_PATH):
    print("Model is not ready yet, app is patiently waiting...")
    time.sleep(5)

st.title("Emotions Prediction")

text = st.text_area("Enter text for emotion prediction:")

if st.button("Predict"):
    if text.strip():
        response = requests.post(API_URL, json={"text": text})
        st.write("Prediction:", response.json()["prediction"])
    else:
        st.warning("Please enter some text!")
