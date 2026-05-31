from datetime import datetime, timedelta
import pandas as pd


from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data.clean import load_raw_data, clean_data, prepare_dataset
from src.data.features import FeatureEngineer


default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def extract_and_clean():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    cleaned.to_csv("/data/cleaned_titanic.csv", index=False)
    print(f"Очищенные данные сохранены: {len(cleaned)} строк")

def prepare_features():
    df = pd.read_csv("/data/cleaned_titanic.csv")
    X, y = prepare_dataset(df)
    fe = FeatureEngineer()
    X_fe = fe.fit_transform(X)
    X_fe.to_csv("/data/features_titanic.csv", index=False)
    y.to_csv("/data/target_titanic.csv", index=False)
    print(f"Признаки подготовлены: {X_fe.shape}")


with DAG(
    "titanic_data_pipeline",
    default_args=default_args,
    description="Очистка и подготовка данных Титаник",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 5, 1),
    catchup=False,
    tags=["titanic", "data"],
) as dag:

    clean_task = PythonOperator(
        task_id="extract_and_clean",
        python_callable=extract_and_clean,
    )

    feature_task = PythonOperator(
        task_id="prepare_features",
        python_callable=prepare_features,
    )

    clean_task >> feature_task
