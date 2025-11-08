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
├── artifacts/               # Trained models
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

### Training
```bash
docker build -t fraud-detection:latest -f Dockerfile-training .
docker run -v $(pwd)/artifacts:/app/artifacts fraud-detection:latest
```

### API
```bash
docker build -t fraud-detection-api:latest -f Dockerfile-inference .
docker run -p 8000:8000 fraud-detection-api:latest
# Visit http://localhost:8000/docs
```

## Tech Stack
- Python 3.13
- scikit-learn, pandas, numpy
- FastAPI for model serving
- Docker for containerization
- joblib for model serialization