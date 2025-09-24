from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import timedelta
from airflow.utils.dates import days_ago
import requests
import time

default_args = {
    'owner': 'ksusha',
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

def health_check(service, url):
    for i in range(10):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service} is healthy!")
                return
        except:
            print(f"Attempt {i+1} for {service} failed")
            time.sleep(5)
    raise Exception(f"{service} failed to start")

with DAG(
    dag_id='deploy_services',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    cleanup = BashOperator(
        task_id='cleanup',
        bash_command="docker rm -f model-api model-app || true"
    )

    build_api = BashOperator(
        task_id='build_api',
        bash_command="docker build --network=host -t model-api -f /opt/airflow/code/deployment/api/Dockerfile.api /opt/airflow/code/deployment/api"
    )

    run_api = BashOperator(
        task_id='run_api',
        # bash_command="docker run -d --name model-api -p 8000:8000 model-api"
        bash_command="docker run -d --name model-api --network airflow_default -p 8000:8000 model-api"
    )

    check_api = PythonOperator(
        task_id='check_api',
        python_callable=health_check,
        # op_kwargs={'service': 'API', 'url': 'http://localhost:8000/health'}
        op_kwargs={'service': 'API', 'url': 'http://model-api:8000/health'}
    )

    build_app = BashOperator(
        task_id='build_app',
        # bash_command="docker build -t model-app -f /opt/airflow/code/deployment/app/Dockerfile.app /opt/airflow/code/deployment/app",
        bash_command="docker run -d --name model-app --network airflow_default -p 8501:8501 model-app"
    )

    run_app = BashOperator(
        task_id='run_app',
        bash_command="docker run -d --name model-app -p 8501:8501 model-app"
    )

    check_app = PythonOperator(
        task_id='check_app',
        python_callable=health_check,
        # op_kwargs={'service': 'App', 'url': 'http://localhost:8501/health'}
        op_kwargs={'service': 'App', 'url': 'http://model-app:8501'}

    )

    cleanup >> build_api >> run_api >> check_api >> build_app >> run_app >> check_app