import streamlit as st
import requests
import time
import os

API_URL = "http://model_api:8000/predict"


# MODEL_PATH = "/opt/airflow/final_model/cool_model.pkl"
MODEL_PATH = "/app/final_model/cool_model.pkl"



while not os.path.exists(MODEL_PATH):
    print("Model is not ready yet, app is patiently waiting...")
    time.sleep(5)

st.title("Emotions Prediction")
st.markdown("""
This app predicts the main emotion in the text you provide.  
Type a sentence or phrase in english, and the model will try to identify the emotion.
""")

SELECTED_EMS = ['anger', 'confusion', 'disgust', 'excitement',
                'fear', 'joy', 'love', 'sadness', 'neutral']

emotion_colors = {
    "anger": "red",
    "confusion": "gray",
    "disgust": "green",
    "excitement": "orange",
    "fear": "purple",
    "joy": "yellow",
    "love": "pink",
    "sadness": "blue",
    "neutral": "lightgray"
}

text = st.text_area("Enter text for emotion prediction:")

# if st.button("Predict"):
#     if text.strip():
#         response = requests.post(API_URL, json={"text": text})
#         st.write("Prediction:", response.json()["prediction"])
#     else:
#         st.warning("Please enter some text!")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text for analysis!")
    else:
        try:
            response = requests.post(API_URL, json={"text": text}, timeout=10)
            data = response.json()

            prediction = data.get("prediction")
            if not prediction:
                st.info("Sorry, we couldn't detect any emotion from the text.")
            else:
                st.markdown("**Predicted emotion:**")
                for emotion in SELECTED_EMS:
                    color = emotion_colors.get(emotion, "black")
                    if emotion == prediction:
                        st.markdown(f"<span style='color:{color}; font-weight:bold'>{emotion}</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color:{color}'>{emotion}</span>", unsafe_allow_html=True)
        except requests.exceptions.RequestException:
            st.error("Failed to connect to the API. Please try again later.")