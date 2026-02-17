"""
MLflow Integration
Experiment tracking, model registry, and versioning
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking and model registry"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self._setup_mlflow()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow_config = self.config.get('mlflow', {})

        # Set tracking URI
        tracking_uri = mlflow_config.get(
            'tracking_uri', 'http://localhost:5000')
        mlflow.set_tracking_uri(tracking_uri)

        # Set registry URI
        registry_uri = mlflow_config.get('registry_uri', 'sqlite:///mlflow.db')
        mlflow.set_registry_uri(registry_uri)

        logger.info(
            f"MLflow configured - Tracking: {tracking_uri}, Registry: {registry_uri}")

    def start_experiment(self, experiment_name: str):
        """Start a new experiment"""
        mlflow.set_experiment(experiment_name)
        logger.info(f"Started experiment: {experiment_name}")

    def log_segmentation_model(self, model, model_name: str,
                               metrics: Dict[str, float], algorithm: str):
        """Log segmentation model"""

        with mlflow.start_run(run_name=f"{algorithm}_segmentation"):
            # Log parameters
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("n_clusters", metrics.get('n_clusters', 0))

            # Log metrics
            mlflow.log_metric("silhouette_score",
                              metrics.get('silhouette_score', 0))
            mlflow.log_metric("davies_bouldin_score",
                              metrics.get('davies_bouldin_score', 0))
            mlflow.log_metric("calinski_harabasz_score",
                              metrics.get('calinski_harabasz_score', 0))

            # Log model
            mlflow.sklearn.log_model(model, artifact_path="segmentation_model")

            logger.info(f"Logged segmentation model: {model_name}")

    def log_churn_model(self, model, model_name: str,
                        metrics: Dict[str, float], model_type: str):
        """Log churn prediction model"""

        with mlflow.start_run(run_name=f"{model_type}_churn"):
            # Log parameters
            mlflow.log_param("model_type", model_type)

            # Log metrics
            mlflow.log_metric("accuracy", metrics.get('accuracy', 0))
            mlflow.log_metric("precision", metrics.get('precision', 0))
            mlflow.log_metric("recall", metrics.get('recall', 0))
            mlflow.log_metric("f1_score", metrics.get('f1', 0))
            mlflow.log_metric("roc_auc", metrics.get('roc_auc', 0))

            # Log model
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(model, artifact_path="churn_model")
            elif model_type == 'lightgbm':
                mlflow.lightgbm.log_model(model, artifact_path="churn_model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="churn_model")

            logger.info(f"Logged churn model: {model_name}")

    def log_recommendation_model(self, model, model_name: str,
                                 metrics: Dict[str, float], algorithm: str):
        """Log recommendation model"""

        with mlflow.start_run(run_name=f"{algorithm}_recommendation"):
            # Log parameters
            mlflow.log_param("algorithm", algorithm)

            # Log metrics
            mlflow.log_metric("rmse", metrics.get('rmse', 0))
            mlflow.log_metric("precision", metrics.get('precision', 0))
            mlflow.log_metric("recall", metrics.get('recall', 0))

            # Log model
            mlflow.sklearn.log_model(
                model, artifact_path="recommendation_model")

            logger.info(f"Logged recommendation model: {model_name}")

    def register_model(self, model_name: str, artifact_path: str, description: str = ""):
        """Register model in model registry"""

        # Get latest run
        experiment = mlflow.get_experiment_by_name(model_name)

        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id])
            if len(runs) > 0:
                latest_run = runs.iloc[0]

                # Register model
                mlflow.register_model(
                    model_uri=f"runs:/{latest_run.run_id}/{artifact_path}",
                    name=model_name
                )

                logger.info(f"Registered model: {model_name}")

    def log_hyperparameter_grid(self, param_grid: Dict[str, list]):
        """Log hyperparameter grid for hyperparameter tuning"""

        with mlflow.start_run(run_name="hyperparameter_tuning"):
            mlflow.log_dict(param_grid, "param_grid.json")
            logger.info(f"Logged hyperparameter grid")


def setup_experiment_tracking(experiment_name: str):
    """Quick setup for experiment tracking"""

    tracker = MLflowTracker()
    tracker.start_experiment(experiment_name)
    return tracker
