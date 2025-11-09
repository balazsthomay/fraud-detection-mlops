# Fraud Detection MLOps Pipeline

End-to-end MLOps pipeline for credit card fraud detection with automated retraining, conditional deployment, and workflow orchestration.

## Dataset
Credit Card Fraud Detection dataset from Kaggle
- 284,807 transactions
- 0.172% fraud rate (492 fraudulent transactions)
- 30 features (28 PCA-transformed + Time + Amount)

## Model Performance
Random Forest Classifier with optimized decision threshold (0.25)
- Precision: 96%
- Recall: 74%
- F1 Score: 0.88

## Architecture

### Training Pipeline
Containerized model training that saves new model candidates with `-new` suffix for safe comparison before deployment.

### Comparison & Deployment
Automated comparison of new vs. existing models based on F1 score. Only deploys models that outperform the current baseline.

### Inference API
FastAPI service for real-time fraud prediction with hot-reload capability for zero-downtime model updates.

### Workflow Orchestration
Prefect-based scheduling for automated retraining and deployment workflows (configurable via cron).

## Project Structure
```
fraud-mlops/
├── src/
│   ├── data.py              # Data loading
│   ├── preprocessing.py     # Feature scaling & splitting
│   ├── models.py            # Model training
│   ├── evaluate.py          # Metrics & threshold optimization
│   ├── utils.py             # Model persistence with versioning
│   ├── train_pipeline.py    # Training pipeline (saves -new artifacts)
│   ├── compare_and_deploy.py # Conditional deployment logic
│   └── workflow.py          # Prefect orchestration
├── api/
│   └── app.py               # FastAPI inference service
├── artifacts/               # Model artifacts (shared volume)
├── data/                    # Dataset
├── Dockerfile-training      # Training container
├── Dockerfile-inference     # API container
├── Dockerfile-workflow      # Orchestration container
└── requirements.txt
```

## Running Locally

### Start Prefect Server
```bash
prefect server start
# Access UI at http://localhost:4200
```

### Run Orchestrated Workflow
```bash
python -m src.workflow
```

### Manual Training & Comparison
```bash
python -m src.train_pipeline
python -m src.compare_and_deploy
```

### Inference API
```bash
uvicorn api.app:app --reload
# Visit http://localhost:8000/docs
```

## Running with Docker

### Build Containers
```bash
docker build -t fraud-detection:latest -f Dockerfile-training .
docker build -t fraud-detection-api:latest -f Dockerfile-inference .
docker build -t fraud-workflow:latest -f Dockerfile-workflow .
```

### Start Orchestrated Pipeline
```bash
# Start Prefect server on host
prefect server start

# Run workflow container (requires Prefect server)
docker run -v $(pwd)/artifacts:/app/artifacts fraud-workflow:latest
```

### Start API
```bash
docker run -v $(pwd)/artifacts:/app/artifacts -p 8000:8000 fraud-detection-api:latest
```

## API Endpoints
- `GET /` - Health check
- `POST /predict` - Fraud prediction (30 features required)
- `POST /reload` - Hot-reload model from artifacts

## Tech Stack
- Python 3.13
- scikit-learn, pandas, numpy
- FastAPI for model serving
- Prefect for workflow orchestration
- Docker for containerization
- joblib for model serialization