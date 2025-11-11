# Fraud Detection MLOps Pipeline

End-to-end MLOps pipeline for credit card fraud detection with automated testing, CI/CD, cloud deployment, and conditional model updates.

## Dataset
Credit Card Fraud Detection dataset from Kaggle
- 284,807 transactions (10k subset for CI/CD testing)
- 0.172% fraud rate (492 fraudulent transactions)
- 30 features (28 PCA-transformed + Time + Amount)

## Model Performance
Random Forest Classifier with optimized decision threshold
- Precision: 96%
- Recall: 74%
- F1 Score: 0.88

## Architecture

### Training Pipeline (Local)
Trains models locally, saves artifacts with `-new` suffix for comparison before deployment.

### Comparison & Deployment
Compares new vs. baseline models by F1 score. Deploys to S3 only when new model outperforms baseline.

### Inference API (AWS ECS)
FastAPI service deployed on AWS ECS, loads models from S3, serves predictions via public endpoint.

### CI/CD Pipeline (GitHub Actions)
- **Test:** Runs integration tests on every PR (mocked AWS calls)
- **Build:** Builds Docker images and pushes to ECR on main branch merges
- **Deploy:** Manual ECS updates (every code change requires manually updating task definition) - - I'll maybe automate it

## Project Structure
```
fraud-mlops/
├── src/
│   ├── train_pipeline.py    # Training (saves -new artifacts)
│   ├── compare_and_deploy.py # F1 comparison + S3 upload
│   └── utils.py             # Artifact persistence
├── api/
│   └── app.py               # FastAPI (loads from S3)
├── tests/
│   ├── test_training.py     # Training integration tests
│   ├── test_inference_api.py # API integration tests
│   └── test_compare_and_deploy.py # Deployment logic tests
├── .github/workflows/
│   └── ci-cd.yml            # GitHub Actions pipeline
├── data/
│   └── creditcard_ci.csv    # 10k-row test dataset
├── Dockerfile-training
├── Dockerfile-inference
└── requirements.txt
```

## Workflow

### Phase 1: Local Training + Cloud Serving
1. Train locally: `python -m src.train_pipeline`
2. Compare models: `python -m src.compare_and_deploy` (auto-uploads to S3 if better)
3. API in ECS auto-reloads from S3

### CI/CD Flow
- **Pull Request:** Runs tests only
- **Merge to main:** Tests → Build images → Push to ECR

## Running Locally

### Training & Deployment
```bash
python -m src.train_pipeline
python -m src.compare_and_deploy
```

### Inference API (Local)
```bash
uvicorn api.app:app --reload
# Visit http://localhost:8000/docs
```

### Testing
```bash
pytest tests/ -v
```

## AWS Deployment

### Prerequisites
- ECR repositories: `fraud-detection-api`
- S3 bucket: `fraud-mlops-artifacts-bt`
- ECS cluster, task definition, and service configured
- IAM role with S3 and ECR permissions

### API Endpoints
- Health: `http://<ecs-public-ip>:8000/`
- Predict: `POST http://<ecs-public-ip>:8000/predict`
- Reload: `POST http://<ecs-public-ip>:8000/reload`

### Example Request
```bash
curl -X POST http://<ecs-ip>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, 1.19, ..., 2.69]}'
```

## Tech Stack
- **ML:** scikit-learn, pandas, numpy
- **API:** FastAPI, uvicorn
- **Infrastructure:** Docker, AWS (ECS, ECR, S3)
- **CI/CD:** GitHub Actions, pytest
- **Storage:** S3 (artifacts), joblib (serialization)