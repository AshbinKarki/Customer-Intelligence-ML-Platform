"""
Utility functions for ML platform
"""
import os
import yaml
import json
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_model(model, model_path: str, metadata: Dict[str, Any] = None):
    """Save model and metadata"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    if metadata:
        meta_path = model_path.replace('.pkl', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)

    logger.info(f"Model saved to {model_path}")


def load_model(model_path: str):
    """Load model from disk"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def load_metadata(meta_path: str) -> Dict[str, Any]:
    """Load model metadata"""
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def train_test_split(X, y=None, test_size=0.2, random_state=42):
    """Custom train-test split to ensure reproducibility"""
    n = len(X)
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    split = int(n * (1 - test_size))

    if y is not None:
        return X.iloc[indices[:split]], X.iloc[indices[split:]], y.iloc[indices[:split]], y.iloc[indices[split:]]
    return X.iloc[indices[:split]], X.iloc[indices[split:]]


def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame = None, method='minmax'):
    """Normalize features using MinMax or StandardScaler"""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns), scaler

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), scaler


def encode_categorical(X_train: dict, X_test: dict = None):
    """Encode categorical features"""
    from sklearn.preprocessing import LabelEncoder

    label_encoders = {}
    X_train_copy = X_train.copy()

    if isinstance(X_train_copy, pd.DataFrame):
        categorical_cols = X_train_copy.select_dtypes(
            include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_train_copy[col] = le.fit_transform(X_train_copy[col].astype(str))
            label_encoders[col] = le

            if X_test is not None:
                X_test[col] = le.transform(X_test[col].astype(str))

    return X_train_copy, X_test, label_encoders


def get_feature_importance(model, feature_names=None, top_k=15):
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None

    indices = np.argsort(importances)[::-1][:top_k]

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    return {
        'features': [feature_names[i] for i in indices],
        'importances': importances[indices].tolist()
    }


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

    return metrics


def create_model_registry():
    """Initialize model registry structure"""
    registry_path = Path('models/registry')
    registry_path.mkdir(parents=True, exist_ok=True)

    registry = {
        'segmentation': {},
        'churn': {},
        'recommendation': {},
        'nlp': {}
    }

    with open(registry_path / 'models.json', 'w') as f:
        json.dump(registry, f, indent=4)

    logger.info("Model registry created")
    return registry


def register_model(model_type: str, model_name: str, model_path: str, metadata: Dict):
    """Register a trained model"""
    registry_path = Path('models/registry/models.json')

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    registry[model_type][model_name] = {
        'path': model_path,
        'created_at': pd.Timestamp.now().isoformat(),
        'metadata': metadata
    }

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4, default=str)

    logger.info(f"Model {model_name} registered in {model_type}")
