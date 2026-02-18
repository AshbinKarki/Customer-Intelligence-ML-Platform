Customer ML Platform 🚀
A production-ready full-stack AI system for customer analytics, segmentation, churn prediction, and product recommendations with API deployment and scalability.

🎯 Features
1. Customer Segmentation (Unsupervised Learning)
K-Means clustering
DBSCAN clustering
Hierarchical clustering
PCA & t-SNE visualization
Cluster evaluation metrics (silhouette, Davies-Bouldin)
2. Churn Prediction (Supervised Learning)
XGBoost, LightGBM, Random Forest models
SMOTE for handling imbalanced data
ROC-AUC, Precision-Recall evaluation
SHAP feature importance explanations
Neural Network variant
3. Recommendation System
Collaborative Filtering (Matrix Factorization)
Content-Based Filtering
Hybrid Model
Neural Collaborative Filtering
Cold-start handling
4. NLP Sentiment Analysis
Customer review sentiment analysis
Topic modeling (LDA)
BERT-based classification
Text preprocessing pipeline
5. Model Serving & Deployment
FastAPI REST API
Real-time + batch predictions
Model versioning with MLflow
Docker containerization
Kubernetes deployment
Auto-retraining pipeline
6. MLOps & Monitoring
MLflow experiment tracking
Airflow pipeline orchestration
Prometheus metrics
Grafana dashboards
Model performance monitoring
📁 Project Structure
customer-ml-platform/
├── data/                           # Raw & processed data
├── src/
│   ├── data/                       # Data loading & preprocessing
│   ├── features/                   # Feature engineering
│   ├── models/                     # ML models (segmentation, churn, recsys, nlp)
│   ├── api/                        # FastAPI application
│   └── utils/                      # Helper utilities
├── notebooks/                      # Jupyter notebooks for exploration
├── tests/                          # Unit & integration tests
├── dashboard/                      # Streamlit dashboard
├── docker/                         # Docker configurations
├── k8s/                            # Kubernetes manifests
├── airflow/                        # Airflow DAGs
└── monitoring/                     # Prometheus & Grafana configs
🚀 Quick Start
1. Setup Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Generate Data
python src/data/generate_data.py
3. Train Models
python notebooks/01_segmentation.py
python notebooks/02_churn_prediction.py
python notebooks/03_recommendation.py
4. Start API Server
python src/api/main.py
5. Launch Dashboard
streamlit run dashboard/app.py
📊 API Endpoints
Predictions
POST /predict/churn - Predict customer churn probability
POST /predict/segment - Assign customer to segment
POST /predict/recommendations - Get product recommendations
Batch Processing
POST /batch/churn-prediction - Bulk churn predictions
POST /batch/segmentation - Bulk customer segmentation
Models
GET /models/list - List all available models
GET /models/{model_id}/metadata - Get model metadata
POST /models/{model_id}/load - Load specific model version
🔧 Configuration
See config.yaml for:

Model hyperparameters
API settings
Database connections
Cloud infrastructure
📈 MLOps Pipeline
Data Ingestion → Raw customer data
Data Pipeline → Preprocessing & feature engineering
Model Training → Train & evaluate multiple models
Model Registry → MLflow artifact store
API Serving → FastAPI + Uvicorn
Monitoring → Prometheus + Grafana
Auto-retraining → Airflow scheduled jobs
Deployment → Kubernetes clusters
☁️ Cloud Deployment
Supports:

AWS (SageMaker, EC2, S3, RDS)
Google Cloud (Vertex AI, Compute Engine, Cloud SQL)
Azure (Azure ML, Container Instances)
📚 Model Performance
Model	Accuracy	AUC-ROC	F1-Score
XGBoost Churn	0.89	0.92	0.85
LightGBM Churn	0.90	0.93	0.86
CF Recommender	MAE: 0.68	NDCG: 0.78	-
🧪 Testing
pytest tests/ -v --cov=src
📖 Documentation
API Documentation
Model Documentation
Deployment Guide
Inference Guide
🤝 Contributing
Create a feature branch
Commit changes
Run tests & linting
Submit pull request
