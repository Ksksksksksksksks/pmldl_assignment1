from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess

default_args = {
    'owner': 'ksusha',
    'start_date': datetime.now(),
    'retry_delay': timedelta(minutes=2),
    'depends_on_past': False,
    'retries': 1,
}

def unpause_dag(dag_id):
    subprocess.run(["airflow", "dags", "unpause", dag_id], check=True)


with DAG(
        dag_id='init_mlops_pipeline',
        default_args=default_args,
        description='Initialize full MLOps pipeline on Airflow startup',
        schedule_interval='@once',
        catchup=False,
        max_active_runs=1
) as dag:
    start = DummyOperator(task_id='start')

    unpause_next = PythonOperator(
        task_id='unpause_goemotions_preprocessing',
        python_callable=unpause_dag,
        op_args=['goemotions_preprocessing'],
    )

    trigger_data_processing = TriggerDagRunOperator(
        task_id='trigger_data_processing',
        trigger_dag_id='goemotions_preprocessing',
    )

    start >> unpause_next>> trigger_data_processing