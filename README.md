# Emotion Recognition ML Pipeline

This project implements an automated ML pipeline for training and deploying an emotion recognition model using **Airflow**, **MLflow**, and **Docker Compose**.

# Emotion Recognition ML Pipeline

This project implements an automated ML pipeline for training and deploying an emotion recognition model using **Airflow**, **MLflow**, and **Docker Compose**.

## Setup

1. Generate secret keys for Airflow:

   ```bash
   # Example (Linux/macOS)
   export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='sqlite:////tmp/airflow.db'  # or your preferred DB
   export AIRFLOW__CORE__FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
   ```

2. Create a `.env` file in the `airflow/` folder next to `docker-compose.yml` and add the keys:

   ```
   AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=your_generated_conn_string
   AIRFLOW__CORE__FERNET_KEY=your_generated_fernet_key
   ```

## How to Run

1. Stop all previously running containers (if any):

   ```bash
   docker compose down
   ```

2. Build and start the project:

   ```bash
   docker compose up --build -d
   ```

3. Wait a few minutes until all services are up and running. You can check it in docker dekstop.

4. Open the Airflow UI at:

   ```
   http://localhost:8080
   ```

   (default login/password: `admin` / `admin`).

5. DAGs will start automatically. You can monitor training and deployment progress directly in Airflow.

6. Once all DAGs are completed, open the app at:

   ```
   http://localhost:8501
   ```

   and start using the emotion recognition tool.

7. You can also open the MLflow UI to explore experiments, models, metrics, and artifacts:

   ```
   http://localhost:5000
   ```

## Project Structure
Initial structure (before running DAGs):
```
.
├── airflow/
│   ├── dags/
│   │   ├── deploy_dag.py
│   │   ├── data_preparation_dag.py
│   │   ├── init_pipeline_dag.py
│   │   └── train_model_dag.py
│   ├── Dockerfile.airflow
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── .env  # to be created with secret keys
├── code/
│   ├── datasets/
│   │   ├── prepare_emotions_datasets_script.py
│   │   └── prepare_emotions_datasets.ipynb
│   ├── deployment/
│   │   ├── api/
│   │   │   ├── model_api.py
│   │   │   ├── Dockerfile
│   │   │   └── requirements_api.txt
│   │   └── app/
│   │       ├── main.py
│   │       ├── requirements_app.txt
│   │       └── Dockerfile
│   └── models/
│       └── train_model.py

```
Folders created automatically after running DAGs:
```
├── data/
│   ├── full_dataset/
│   └── processed/
│       ├── train.tsv
│       ├── test.tsv
│       └── val.tsv
├── mlruns/          # MLflow experiments, runs, and final model
├── final_model/     # Final tested model for API predictions

---

Note: mlruns/ will contain all training runs and the final model, which will later be saved to final_model/ for deployment via API.
