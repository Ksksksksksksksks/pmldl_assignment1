import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow:5000")

DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data/processed")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/airflow/mlruns")
MLFLOW_MODEL_DIR = os.path.join(MODEL_DIR, "model")
os.makedirs(MLFLOW_MODEL_DIR, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "train.tsv")
TEST_FILE = os.path.join(DATA_DIR, "test.tsv")
SELECTED_EMS = ['anger', 'confusion', 'disgust', 'excitement',
                'fear', 'joy', 'love', 'sadness', 'neutral']

MODEL_PATH = os.path.join(MODEL_DIR, "goemotions_model.pkl")
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "goemotions_model.pkl")

train_df = pd.read_csv(TRAIN_FILE, sep="\t")
test_df = pd.read_csv(TEST_FILE, sep="\t")

X_train_text = train_df['text']
X_test_text = test_df['text']

y_train = train_df[SELECTED_EMS]
y_test = test_df[SELECTED_EMS]

vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

mlflow.set_experiment("goemotions_multi_label")
with mlflow.start_run():
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-weighted: {f1}")

    mlflow.log_metric("f1_weighted", f1)

    mlflow.sklearn.log_model(model, artifact_path="model")   # не нужно mlflow.log_artifact(LOCAL_MODEL_PATH)

    joblib.dump({"model": model, "vectorizer": vectorizer}, LOCAL_MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
