"""
Batch Processing Pipeline
Scheduled batch predictions and model retraining
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from typing import List
import yaml

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch prediction and processing pipeline"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        os.makedirs('data/predictions', exist_ok=True)

    def _load_config(self, config_path: str):
        """Load configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def batch_churn_prediction(self, input_file: str, output_dir: str = 'data/predictions'):
        """Process batch churn predictions"""

        logger.info(f"Starting batch churn prediction on {input_file}")

        # Load data
        df = pd.read_csv(input_file)

        # Feature preprocessing
        feature_cols = ['total_purchases', 'avg_order_value', 'days_active',
                        'login_frequency', 'support_tickets', 'conversion_rate']

        X = df[feature_cols].fillna(df[feature_cols].mean())

        # Make predictions (mock)
        predictions = []
        for idx, row in X.iterrows():
            risk_score = min(
                0.5 * (1 / (1 + np.exp(-(-row['total_purchases'] + 10) / 5))) +
                0.3 * (row['support_tickets'] / 10) +
                0.2 * (1 - row['conversion_rate']),
                1.0
            )
            predictions.append({
                'customer_id': df.iloc[idx]['customer_id'],
                'churn_probability': risk_score,
                'churn_prediction': 1 if risk_score > 0.5 else 0,
                'risk_level': 'HIGH' if risk_score > 0.7 else ('MEDIUM' if risk_score > 0.4 else 'LOW')
            })

        result_df = pd.DataFrame(predictions)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(
            output_dir, f'churn_predictions_{timestamp}.csv')
        result_df.to_csv(output_file, index=False)

        logger.info(f"Batch churn prediction completed")
        logger.info(
            f"High risk customers: {(result_df['churn_probability'] > 0.5).sum()}")
        logger.info(f"Results saved to {output_file}")

        return result_df

    def batch_segmentation(self, input_file: str, output_dir: str = 'data/predictions'):
        """Process batch customer segmentation"""

        logger.info(f"Starting batch segmentation on {input_file}")

        # Load data
        df = pd.read_csv(input_file)

        # Calculate CLV
        df['customer_lifetime_value'] = df['total_purchases'] * \
            df['avg_order_value']

        # Segmentation based on CLV
        def assign_segment(clv):
            if clv > 1000:
                return 4  # Premium
            elif clv > 500:
                return 3  # High Value
            elif clv > 200:
                return 2  # Medium Value
            elif clv > 50:
                return 1  # Low Value
            else:
                return 0  # Inactive

        result_df = df[['customer_id']].copy()
        result_df['segment'] = df['customer_lifetime_value'].apply(
            assign_segment)
        result_df['segment_name'] = result_df['segment'].map({
            0: 'Inactive',
            1: 'Low Value',
            2: 'Medium Value',
            3: 'High Value',
            4: 'Premium'
        })

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'segmentation_{timestamp}.csv')
        result_df.to_csv(output_file, index=False)

        logger.info(f"Batch segmentation completed")
        logger.info(
            f"Segment distribution:\n{result_df['segment_name'].value_counts()}")
        logger.info(f"Results saved to {output_file}")

        return result_df

    def batch_recommendations(self, input_file: str, n_recommendations: int = 5,
                              output_dir: str = 'data/predictions'):
        """Process batch product recommendations"""

        logger.info(f"Starting batch recommendations on {input_file}")

        # Load data
        df = pd.read_csv(input_file)

        recommendations = []
        for idx, row in df.iterrows():
            customer_id = row['customer_id']
            np.random.seed(customer_id)

            # Mock recommendations
            products = np.random.choice(
                range(1, 101), n_recommendations, replace=False)
            scores = np.random.uniform(0.5, 1.0, n_recommendations)

            for product, score in zip(products, scores):
                recommendations.append({
                    'customer_id': customer_id,
                    'product_id': product,
                    'recommendation_score': score
                })

        result_df = pd.DataFrame(recommendations)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(
            output_dir, f'recommendations_{timestamp}.csv')
        result_df.to_csv(output_file, index=False)

        logger.info(f"Batch recommendations completed")
        logger.info(f"Generated {len(result_df)} recommendations")
        logger.info(f"Results saved to {output_file}")

        return result_df

    def generate_analytics_report(self, customers_file: str,
                                  output_dir: str = 'data/reports'):
        """Generate comprehensive analytics report"""

        logger.info("Generating analytics report")

        # Load data
        df = pd.read_csv(customers_file)

        # Calculate metrics
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_customers': len(df),
            'churn_rate': (df['churned'].sum() / len(df) if 'churned' in df else 0),
            'avg_clv': df['total_purchases'].mean() * df['avg_order_value'].mean() if all(col in df for col in ['total_purchases', 'avg_order_value']) else 0,
            'active_customers': (df['days_active'] > 30).sum() if 'days_active' in df else 0,
            'avg_purchase_frequency': df['total_purchases'].mean() if 'total_purchases' in df else 0
        }

        # Save report
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(
            output_dir, f'analytics_report_{timestamp}.json')

        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)

        logger.info(f"Report saved to {report_file}")

        return report


def schedule_batch_jobs():
    """Schedule batch jobs (to be used with Airflow or APScheduler)"""

    logger.info("Setting up batch job scheduler")

    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    processor = BatchProcessor()

    # Schedule daily churn predictions
    scheduler.add_job(
        lambda: processor.batch_churn_prediction('data/raw/customers.csv'),
        'cron',
        hour=2,  # Run at 2 AM
        id='daily_churn_prediction'
    )

    # Schedule daily segmentation
    scheduler.add_job(
        lambda: processor.batch_segmentation('data/raw/customers.csv'),
        'cron',
        hour=3,  # Run at 3 AM
        id='daily_segmentation'
    )

    # Schedule weekly analytics report
    scheduler.add_job(
        lambda: processor.generate_analytics_report('data/raw/customers.csv'),
        'cron',
        day_of_week=6,  # Sunday
        hour=4,
        id='weekly_analytics'
    )

    scheduler.start()
    logger.info("Batch job scheduler started")

    return scheduler
