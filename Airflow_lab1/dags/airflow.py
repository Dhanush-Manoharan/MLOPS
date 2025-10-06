from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

default_args = {
    "owner": "student",
    "start_date": datetime(2025, 10, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="bank_marketing_etl_kmeans",
    description="ETL + KMeans elbow on UCI Bank Marketing (bank-full.csv)",
    default_args=default_args,
    schedule_interval=None,    # ← fixed
    catchup=False,
    params={"source_path": "/opt/airflow/data/bank-full.csv"},
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
        op_kwargs={"source_path": "{{ params.source_path }}"},
    )

    preprocess_task = PythonOperator(
        task_id="data_preprocessing",
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    build_save_model_task = PythonOperator(
        task_id="build_save_model",
        python_callable=build_save_model,
        op_args=[preprocess_task.output, "/opt/airflow/working_data/model_bank.sav"],
    )

    elbow_task = PythonOperator(
        task_id="load_model_elbow",
        python_callable=load_model_elbow,
        op_args=["/opt/airflow/working_data/model_bank.sav", build_save_model_task.output],
    )

    load_data_task >> preprocess_task >> build_save_model_task >> elbow_task
