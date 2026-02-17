"""
Airflow DAG for Customer ML Platform
Orchestrates data processing, model training, and batch predictions
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    'owner': 'ml-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# Define DAG
dag = DAG(
    'customer_ml_pipeline',
    default_args=default_args,
    description='Customer ML Platform - Data Processing & Model Training',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    catchup=False,
)


def data_ingestion_task():
    """Load and validate customer data"""
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting data ingestion...")

    # Load data
    customers = pd.read_csv('data/raw/customers.csv')
    purchases = pd.read_csv('data/raw/purchase_history.csv')

    logger.info(
        f"Loaded {len(customers)} customers and {len(purchases)} purchases")


def feature_engineering_task():
    """Engineer features for modeling"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting feature engineering...")


def train_churn_model_task():
    """Train churn prediction models"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Training churn prediction models...")


def train_segmentation_model_task():
    """Train customer segmentation models"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Training segmentation models...")


def batch_predictions_task():
    """Generate batch predictions"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Generating batch predictions...")


def model_evaluation_task():
    """Evaluate model performance"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model performance...")


def model_registry_task():
    """Register models in MLflow"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Registering models in MLflow...")


# Define tasks
task_data_ingestion = PythonOperator(
    task_id='data_ingestion',
    python_callable=data_ingestion_task,
    dag=dag,
)

task_feature_engineering = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering_task,
    dag=dag,
)

task_train_churn = PythonOperator(
    task_id='train_churn_model',
    python_callable=train_churn_model_task,
    dag=dag,
)

task_train_segmentation = PythonOperator(
    task_id='train_segmentation_model',
    python_callable=train_segmentation_model_task,
    dag=dag,
)

task_batch_predictions = PythonOperator(
    task_id='batch_predictions',
    python_callable=batch_predictions_task,
    dag=dag,
)

task_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation_task,
    dag=dag,
)

task_registry = PythonOperator(
    task_id='model_registry',
    python_callable=model_registry_task,
    dag=dag,
)

# Define dependencies
task_data_ingestion >> task_feature_engineering
task_feature_engineering >> [task_train_churn, task_train_segmentation]
[task_train_churn, task_train_segmentation] >> task_batch_predictions
task_batch_predictions >> task_evaluation
task_evaluation >> task_registry
