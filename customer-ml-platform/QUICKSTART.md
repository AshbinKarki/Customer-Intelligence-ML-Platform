# Getting Started Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data
```bash
python src/data/generate_data.py
```

### 3. Run Training Notebook
Open and run `notebooks/customer_ml_comprehensive.ipynb` to:
- Load and analyze customer data
- Engineer features
- Train clustering models (K-Means, DBSCAN, Hierarchical)
- Build churn prediction models (XGBoost, LightGBM, RF)
- Create recommendation system (Collaborative Filtering)
- Perform NLP sentiment analysis
- Deploy models with FastAPI

### 4. Start API Server
```bash
python src/api/main.py
```
API will be available at `http://localhost:8000`

### 5. Launch Dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard at `http://localhost:8501`

---

## Docker Deployment

### Build Images
```bash
cd docker
docker build -t customer-ml-api:latest --target api -f Dockerfile ..
docker build -t customer-ml-dashboard:latest --target dashboard -f Dockerfile ..
```

### Run with Docker Compose
```bash
cd docker
docker-compose up -d
```

Services:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## Kubernetes Deployment

### Prerequisites
- Docker images pushed to registry
- kubectl configured
- Kubernetes cluster running

### Deploy
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/airflow.yaml
```

### Check Status
```bash
kubectl get pods -n customer-ml
kubectl logs -n customer-ml deployment/customer-ml-api
```

### Access Services
```bash
kubectl port-forward -n customer-ml svc/customer-ml-api 8000:8000
kubectl port-forward -n customer-ml svc/customer-ml-dashboard 8501:8501
```

---

## API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict Churn
```bash
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 123,
    "total_purchases": 15,
    "avg_order_value": 50.0,
    "days_active": 180,
    "login_frequency": 20,
    "support_tickets": 2,
    "conversion_rate": 0.3,
    "subscription_tier": "premium",
    "device_type": "mobile",
    "country": "US"
  }'
```

### Get Recommendations
```bash
curl -X POST http://localhost:8000/predict/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 123,
    "last_purchased_product": 42,
    "n_recommendations": 5
  }'
```

### Batch Predictions
```bash
curl -X POST http://localhost:8000/predict/churn/batch \
  -F "file=@data/raw/customers_batch.csv"
```

---

## Project Structure

```
customer-ml-platform/
├── src/
│   ├── api/                 # FastAPI application
│   ├── models/              # ML models (segmentation, churn, recsys, nlp)
│   ├── data/                # Data loading & preprocessing
│   ├── features/            # Feature engineering
│   └── utils/               # Helpers & utilities
├── notebooks/               # Jupyter notebooks
├── dashboard/               # Streamlit dashboard
├── docker/                  # Docker configuration
├── k8s/                     # Kubernetes manifests
├── airflow/                 # Airflow DAGs
├── monitoring/              # Prometheus & Grafana config
├── tests/                   # Unit & integration tests
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Configuration

Edit `config.yaml` for:
- Model hyperparameters
- API settings
- Database connections
- Cloud provider settings
- MLflow configuration

---

## Monitoring & Alerts

### Prometheus Metrics
Access at `http://localhost:9090`

### Grafana Dashboards
Access at `http://localhost:3000` (admin/admin)

### MLflow Tracking
Access at `http://localhost:5000`

---

## Common Issues & Solutions

### Port Already in Use
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Database Connection Error
Ensure PostgreSQL is running:
```bash
docker ps  # Should show customer-ml-postgres container
```

### Memory Issues
Reduce batch sizes in `config.yaml`:
```yaml
inference:
  batch_size: 500  # Reduced from 1000
```

---

## Next Steps

1. **Fine-tune models** with your own customer data
2. **Setup continuous training** with Airflow DAGs
3. **Deploy to cloud** (AWS/GCP/Azure)
4. **Setup monitoring** with custom metrics
5. **Implement A/B testing** for recommendation changes
6. **Scale with Kubernetes** for production workloads

---

For more details, see:
- [API Documentation](./docs/api.md)
- [Model Documentation](./docs/models.md)
- [Deployment Guide](./docs/deployment.md)
