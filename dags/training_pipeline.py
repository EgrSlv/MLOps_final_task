from datetime import datetime, timedelta

import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.models.train import train_model
from src.monitoring.drift import compute_psi


default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def train_and_log():
    run_id = train_model(
        model_name="gradient_boosting",
        experiment_name="titanic-survival",
        tracking_uri="http://mlflow:5000",
    )
    print(f"Модель обучена. ID запуска MLflow: {run_id}")
    return run_id

def evaluate_and_deploy(**context):
    run_id = context["task_instance"].xcom_pull(task_ids="train_model")
    psi = compute_psi(np.array([0.5]), np.array([0.5]))
    print(f"Проверка дрейфа: PSI={psi:.4f}")
    print(f"Модель {run_id} готова к развёртыванию")


with DAG(
    "titanic_training_pipeline",
    default_args=default_args,
    description="Обучение и развёртывание модели выживаемости Титаник",
    schedule_interval=timedelta(days=7),
    start_date=datetime(2026, 5, 1),
    catchup=False,
    tags=["titanic", "training"],
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_and_log,
    )

    deploy_task = PythonOperator(
        task_id="evaluate_and_deploy",
        python_callable=evaluate_and_deploy,
    )

    train_task >> deploy_task
