import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib

mlflow.set_tracking_uri("http://mlflow:5000")

DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data/processed")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/airflow/mlruns")
MLFLOW_MODEL_DIR = os.path.join(MODEL_DIR, "model")
os.makedirs(MLFLOW_MODEL_DIR, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "train.tsv")
TEST_FILE = os.path.join(DATA_DIR, "test.tsv")


print("=== DEBUG: Paths ===")
print("DATA_DIR:", DATA_DIR)
print("train_path:", TRAIN_FILE)
print("test_path:", TEST_FILE)
print("====================")


SELECTED_EMS = ['anger', 'confusion', 'disgust', 'excitement',
                'fear', 'joy', 'love', 'sadness', 'neutral']

MODEL_PATH = os.path.join(MODEL_DIR, "goemotions_model.pkl")
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "goemotions_model.pkl")

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"train.tsv not found at {TRAIN_FILE}")
if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(f"test.tsv not found at {TEST_FILE}")

print("=== DEBUG: Loading data ===")

train_df = pd.read_csv(TRAIN_FILE, sep="\t")
test_df = pd.read_csv(TEST_FILE, sep="\t")

print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)
print("train_df columns:", train_df.columns.tolist())
print("test_df columns:", test_df.columns.tolist())

missing_cols = [c for c in SELECTED_EMS if c not in train_df.columns]
if missing_cols:
    raise KeyError(f"Missing expected label columns in train_df: {missing_cols}")


X_train_text = train_df['text']
X_test_text = test_df['text']

y_train = train_df[SELECTED_EMS]
y_test = test_df[SELECTED_EMS]

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

EXPERIMENT_NAME = "goemotions_multi_label"
ARTIFACT_LOCATION = "/opt/airflow/mlruns"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("clf", MultiOutputClassifier(LogisticRegression(max_iter=1000)))
    ])

    print("=== DEBUG: Training model ===")
    pipeline.fit(X_train_text, y_train)

    print("=== DEBUG: Predicting ===")
    y_pred = pipeline.predict(X_test_text)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-weighted: {f1}")

    print("=== DEBUG: Logging to MLflow ===")
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("f1_weighted", f1)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="goemotions_pipeline"
    )
    mlflow.log_artifact("goemotions_model.pkl")

    LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "goemotions_model.pkl")
    joblib.dump(pipeline, LOCAL_MODEL_PATH)

    print("=== DEBUG: Finished successfully ===")
