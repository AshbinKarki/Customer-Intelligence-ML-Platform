"""
Feature Engineering Pipeline
Transforms raw customer data into ML-ready features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List


class FeatureEngineer:
    """Feature engineering for customer ML platform"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw customer data"""
        df = df.copy()

        # Temporal features
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        df['customer_age_days'] = (
            pd.Timestamp.now() - df['signup_date']).dt.days

        # Purchase behavior features
        df['purchase_frequency'] = df['total_purchases'] / \
            (df['customer_age_days'] + 1)
        df['revenue_per_day'] = df['customer_lifetime_value'] / \
            (df['customer_age_days'] + 1)

        # Engagement features
        df['engagement_score'] = (
            (df['login_frequency'] / (df['login_frequency'].max() + 1)) * 0.4 +
            (df['conversion_rate']) * 0.4 +
            (1 - df['support_tickets'] /
             (df['support_tickets'].max() + 1)) * 0.2
        )

        # Risk features (churn indicators)
        df['risk_score'] = (
            (1 - df['conversion_rate']) * 0.3 +
            (df['support_tickets'] / (df['support_tickets'].max() + 1)) * 0.3 +
            (df['days_since_last_purchase'] /
             (df['days_since_last_purchase'].max() + 1)) * 0.4
        )

        # Spending patterns
        df['high_value_customer'] = (
            df['customer_lifetime_value'] > df['customer_lifetime_value'].quantile(0.75)).astype(int)
        df['active_customer'] = (
            df['days_since_last_purchase'] < 30).astype(int)

        # Subscription value
        tier_values = {'free': 0, 'basic': 1, 'premium': 2, 'enterprise': 3}
        df['subscription_value'] = df['subscription_tier'].map(tier_values)

        return df

    def prepare_for_modeling(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for model training"""

        # Encode categorical variables
        categorical_cols = ['country', 'device_type', 'subscription_tier']
        df_encoded = df.copy()

        for col in categorical_cols:
            if fit_scaler:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
            else:
                df_encoded[col] = self.label_encoders[col].transform(
                    df_encoded[col])

        # Select features for modeling
        feature_cols = [
            'total_purchases', 'avg_order_value', 'days_active', 'login_frequency',
            'support_tickets', 'conversion_rate', 'customer_lifetime_value',
            'avg_session_duration', 'customer_age_days', 'purchase_frequency',
            'revenue_per_day', 'engagement_score', 'risk_score',
            'high_value_customer', 'active_customer', 'subscription_value',
            'country', 'device_type', 'subscription_tier'
        ]

        X = df_encoded[feature_cols].fillna(0)

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        self.feature_names = feature_cols

        return X_scaled, feature_cols

    def get_segmentation_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """Get features optimized for clustering"""

        feature_cols = [
            'total_purchases', 'avg_order_value', 'customer_lifetime_value',
            'login_frequency', 'conversion_rate', 'engagement_score'
        ]

        X = df[feature_cols].fillna(0)

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled


def apply_pca(X_train, X_test=None, n_components=2):
    """Apply PCA for dimensionality reduction and visualization"""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)

    result = {
        'X_train': X_train_pca,
        'pca': pca,
        'explained_variance': pca.explained_variance_ratio_.sum()
    }

    if X_test is not None:
        X_test_pca = pca.transform(X_test)
        result['X_test'] = X_test_pca

    return result


def apply_tsne(X, n_components=2, perplexity=30):
    """Apply t-SNE for visualization"""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=n_components,
                perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    return X_tsne
