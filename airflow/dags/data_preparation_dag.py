from airflow import DAG
from airflow.operators.bash import BashOperator
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

    # Скачивание данных
    download_data = BashOperator(
        task_id='download_data',
        bash_command=(
            'mkdir -p ~/project/data/full_dataset && '
            'wget -P ~/project/data/full_dataset https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv && '
            'wget -P ~/project/data/full_dataset https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv && '
            'wget -P ~/project/data/full_dataset https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv'
        )
    )

    preprocess = BashOperator(
        task_id='preprocess_data',
        bash_command='jupyter nbconvert --to notebook --execute ~/project/notebooks/preprocess.ipynb --output ~/project/notebooks/preprocess_out.ipynb'
    )

    download_data >> preprocess
