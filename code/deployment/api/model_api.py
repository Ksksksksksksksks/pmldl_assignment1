from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import time

app = FastAPI(title="GoEmotions API")

# MODEL_PATH = "/opt/airflow/final_model/cool_model.pkl"
MODEL_PATH = "cool_model.pkl"


while not os.path.exists(MODEL_PATH):
    print("Model is not ready yet, api is patiently waiting...")
    time.sleep(5)

model = joblib.load(MODEL_PATH)

class TextInput(BaseModel):
    text: str

SELECTED_EMS = ['anger', 'confusion', 'disgust', 'excitement',
                'fear', 'joy', 'love', 'sadness', 'neutral']

@app.get("/")
def root():
    return {"message": "GoEmotions API is running. Use POST /predict with {'text': 'your text'}"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: TextInput):
    pred_vector = model.predict([input.text])[0]
    pred_labels = [label for label, val in zip(SELECTED_EMS, pred_vector) if val == 1]
    return {"prediction": pred_labels}

