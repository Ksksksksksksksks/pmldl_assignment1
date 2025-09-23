from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ksusha',
    'start_date': datetime(2025, 9, 20),
    'retry_delay': timedelta(minutes=2),
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
        dag_id='init_mlops_pipeline',
        default_args=default_args,
        description='Initialize full MLOps pipeline on Airflow startup',
        schedule_interval='@once',
        catchup=False,
        max_active_runs=1
) as dag:
    start = DummyOperator(task_id='start')

    trigger_data_processing = TriggerDagRunOperator(
        task_id='trigger_data_processing',
        trigger_dag_id='goemotions_preprocessing',
    )

    start >> trigger_data_processing