# Fraud Detection MLOps Pipeline

MLOps pipeline for credit card fraud detection.

## Dataset
Credit Card Fraud Detection dataset from Kaggle
- 284,807 transactions
- 0.172% fraud rate (492 fraudulent transactions)
- 30 features (28 PCA-transformed + Time + Amount)

## Model Performance
Random Forest Classifier with optimized decision threshold (0.25)
- Precision: 92%
- Recall: 85%
- F1 Score: 0.88

## Architecture

### Training Pipeline
Containerized model training with automatic artifact persistence and API notification.

### Inference API
FastAPI service for real-time fraud prediction with hot-reload capability for model updates.

### Model Update Flow
1. Training container retrains model and saves artifacts to shared volume
2. Training notifies inference API via HTTP POST to `/reload`
3. API reloads model in-memory without downtime

## Project Structure
```
fraud-mlops/
├── src/
│   ├── data.py              # Data loading
│   ├── preprocessing.py     # Feature scaling & splitting
│   ├── models.py            # Model training
│   ├── evaluate.py          # Metrics & threshold optimization
│   ├── visualize.py         # Performance plots
│   ├── utils.py             # Model persistence
│   └── train_pipeline.py    # End-to-end training
├── api/
│   └── app.py               # FastAPI inference service
├── artifacts/               # Trained models (shared volume)
├── data/                    # Dataset
├── Dockerfile-training      # Training container
├── Dockerfile-inference     # API container
└── requirements.txt
```

## Running Locally

### Training
```bash
python -m src.train_pipeline
```

### API
```bash
uvicorn api.app:app --reload
# Visit http://localhost:8000/docs
```

## Running with Docker

### Start API (with volume mount for live model updates)
```bash
docker build -t fraud-detection-api:latest -f Dockerfile-inference .
docker run -v $(pwd)/artifacts:/app/artifacts -p 8000:8000 fraud-detection-api:latest
```

### Train Model (automatically notifies API to reload)
```bash
docker build -t fraud-detection:latest -f Dockerfile-training .
docker run -v $(pwd)/artifacts:/app/artifacts fraud-detection:latest
```

### API Endpoints
- `GET /` - Health check
- `POST /predict` - Fraud prediction (requires 30 features)
- `POST /reload` - Hot-reload model from artifacts

## Tech Stack
- Python 3.13
- scikit-learn, pandas, numpy
- FastAPI for model serving
- Docker for containerization
- joblib for model serialization