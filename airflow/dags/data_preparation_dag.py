from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'ksusha',
    'start_date': days_ago(1),
}

with DAG(
    dag_id='goemotions_preprocessing',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    install_deps = BashOperator(
        task_id='install_dependencies',
        bash_command='pip install iterative-stratification'
    )

    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='cd /opt/airflow/code/datasets && python prepare_emotions_datasets_script.py --data_dir /opt/airflow/data/full_dataset --output_dir /opt/airflow/data/processed'
    )

    # download_data >> preprocess_data
    install_deps >> preprocess_data