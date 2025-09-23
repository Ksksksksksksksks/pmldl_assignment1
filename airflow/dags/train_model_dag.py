from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import subprocess
import os
import requests

default_args = {'owner': 'ksusha', 'start_date': days_ago(1)}

MODEL_PATH = "/opt/airflow/mlruns/goemotions_model.pkl"
MLFLOW_URI = "http://mlflow:5000"

def check_mlflow_task():
    try:
        r = requests.get(MLFLOW_URI)
        if r.status_code != 200:
            raise ConnectionError(f"MLflow returned status {r.status_code}")
    except Exception as e:
        raise ConnectionError(f"Cannot connect to MLflow at {MLFLOW_URI}: {e}")


def train_model_task():
    subprocess.run(
        ["python", "/opt/airflow/code/models/train_model.py"],
        check=True
    )

def push_to_dvc_task():
    if os.path.exists(MODEL_PATH):
        subprocess.run(["dvc", "add", MODEL_PATH], check=True, cwd="/opt/airflow")
        subprocess.run(["dvc", "push"], check=True, cwd="/opt/airflow")
    else:
        raise FileNotFoundError(f"Model not found here: {MODEL_PATH}")


with DAG(
    dag_id='goemotions_train_model',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:
    check_mlflow = PythonOperator(
        task_id="check_mlflow",
        python_callable=check_mlflow_task
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task
    )

    push_to_dvc = PythonOperator(
        task_id="push_to_dvc",
        python_callable=push_to_dvc_task
    )

    check_mlflow >> train_model >> push_to_dvc
