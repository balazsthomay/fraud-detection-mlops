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
├── artifacts/               # Trained models (mounted volume)
├── data/                    # Dataset
├── Dockerfile               # Training container
└── requirements.txt
```

## Running the Training Pipeline

### Local
```bash
python -m src.train_pipeline
```

### Docker
```bash
# Build image
docker build -t fraud-detection:latest .

# Train model (saves artifacts to local directory)
docker run -v $(pwd)/artifacts:/app/artifacts fraud-detection:latest
```

## Tech Stack
- Python 3.13
- scikit-learn, pandas, numpy
- Docker for containerization
- joblib for model serialization