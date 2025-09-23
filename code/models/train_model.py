import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib
from itertools import product

mlflow.set_tracking_uri("http://mlflow:5000")

DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data/processed")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/airflow/mlruns")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "train.tsv")
VAL_FILE = os.path.join(DATA_DIR, "val.tsv")
TEST_FILE = os.path.join(DATA_DIR, "test.tsv")

SELECTED_EMS = ['anger', 'confusion','disgust',
                    'excitement', 'fear', 'joy',
                    'love', 'sadness', 'surprise','neutral']

train_df = pd.read_csv(TRAIN_FILE, sep="\t")
val_df = pd.read_csv(VAL_FILE, sep="\t")
test_df = pd.read_csv(TEST_FILE, sep="\t")

X_train_text = train_df['text']
y_train = train_df[SELECTED_EMS]

X_val_text = val_df['text']
y_val = val_df[SELECTED_EMS]

X_test_text = test_df['text']
y_test = test_df[SELECTED_EMS]

param_grid = {
    "max_features": [20000, 30000],
    "ngram_range": [(1,1), (1,2)],
    "C": [0.5, 1.0]
}

EXPERIMENT_NAME = "goemotions_hyperparam_tuning"
mlflow.set_experiment(EXPERIMENT_NAME)

best_f1 = 0
best_pipeline = None
best_thresholds = None
best_params = None

for max_features, ngram_range, C in product(param_grid["max_features"],
                                            param_grid["ngram_range"],
                                            param_grid["C"]):
    with mlflow.start_run():
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=True,
                strip_accents='unicode',
                min_df=2,
                max_df=0.95
            )),
            ("clf", MultiOutputClassifier(
                LogisticRegression(
                    max_iter=3000,
                    class_weight='balanced',
                    C=C,
                    solver='liblinear'
                )
            ))
        ])

        pipeline.fit(X_train_text, y_train)

        y_val_proba = pipeline.predict_proba(X_val_text)
        thresholds = {}
        y_val_pred = pd.DataFrame()
        for i, em in enumerate(SELECTED_EMS):
            probs = y_val_proba[i][:,1]
            best_em_f1 = 0
            best_thresh = 0.5
            for t in np.arange(0.1, 0.9, 0.05):
                pred = (probs >= t).astype(int)
                f1 = f1_score(y_val[em], pred)
                if f1 > best_em_f1:
                    best_em_f1 = f1
                    best_thresh = t
            thresholds[em] = best_thresh
            y_val_pred[em] = (probs >= best_thresh).astype(int)

        f1_macro_val = f1_score(y_val, y_val_pred, average='macro')

        mlflow.log_params({
            "max_features": max_features,
            "ngram_range": str(ngram_range),
            "C": C
        })
        mlflow.log_metrics({
            "f1_macro_val": f1_macro_val
        })

        if f1_macro_val > best_f1:
            best_f1 = f1_macro_val
            best_pipeline = pipeline
            best_thresholds = thresholds
            best_params = {"max_features": max_features, "ngram_range": ngram_range, "C": C}

            mlflow.sklearn.log_model(best_pipeline, "best_pipeline")

print("Best params:", best_params)
print("Best thresholds:", best_thresholds)
print("Best F1-macro on val:", best_f1)

X_trainval_text = pd.concat([X_train_text, X_val_text])
y_trainval = pd.concat([y_train, y_val])
best_pipeline.fit(X_trainval_text, y_trainval)

y_test_proba = best_pipeline.predict_proba(X_test_text)
y_test_pred = pd.DataFrame()
for i, em in enumerate(SELECTED_EMS):
    thresh = best_thresholds[em]
    y_test_pred[em] = (y_test_proba[i][:,1] >= thresh).astype(int)

f1_weighted_test = f1_score(y_test, y_test_pred, average='weighted')
f1_macro_test = f1_score(y_test, y_test_pred, average='macro')
f1_micro_test = f1_score(y_test, y_test_pred, average='micro')

print(f"Test F1-weighted: {f1_weighted_test}, F1-macro: {f1_macro_test}, F1-micro: {f1_micro_test}")

FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "goemotions_final_model.pkl")
joblib.dump(best_pipeline, FINAL_MODEL_PATH)
