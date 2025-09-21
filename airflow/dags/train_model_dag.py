from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import subprocess

default_args = {'owner': 'ksusha', 'start_date': days_ago(1)}

MODEL_PATH = "/opt/airflow/code/models/goemotions_model.pkl"

def train_model_task():
    subprocess.run(
        ["python", "/opt/airflow/code/models/train_model.py"],
        check=True
    )

def push_to_dvc_task():
    subprocess.run(
        ["dvc", "add", MODEL_PATH],
        check=True
    )
    subprocess.run(
        ["dvc", "push"],
        check=True
    )

with DAG(
    dag_id='goemotions_train_model',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task
    )

    push_to_dvc = PythonOperator(
        task_id="push_to_dvc",
        python_callable=push_to_dvc_task
    )

    train_model >> push_to_dvc
