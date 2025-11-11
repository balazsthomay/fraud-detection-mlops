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

## Architecture Overview
```
Local Training → Compare F1 → Upload to S3 → ECS API reloads → Serves predictions
                                    ↓
                          Code changes → GitHub Actions → ECR → ECS deployment
```

### Training Pipeline (Local)
Trains models locally with full dataset, saves artifacts with `-new` suffix for safe comparison before deployment.

### Comparison & Deployment
Automated comparison of new vs. baseline models by F1 score. Uploads to S3 and notifies API only when new model outperforms baseline.

### Inference API (AWS ECS)
FastAPI service running on AWS ECS Fargate, downloads models from S3 on startup and reload, serves predictions via public endpoint with zero-downtime updates.

### CI/CD Pipeline (GitHub Actions)
- **Test (on PRs):** Integration tests with mocked AWS calls
- **Build (on main):** Docker images pushed to ECR
- **Deploy (on main):** Automated ECS task definition updates and service deployment

## Project Structure
```
fraud-mlops/
├── src/
│   ├── train_pipeline.py         # Training with -new suffix
│   ├── compare_and_deploy.py     # F1 comparison + S3 upload + API notification
│   ├── utils.py                  # Artifact persistence with versioning
│   ├── data.py, preprocessing.py, models.py, evaluate.py
├── api/
│   └── app.py                    # FastAPI with S3 integration
├── tests/
│   ├── test_training.py          # Training integration tests
│   ├── test_inference_api.py     # API integration tests (mocked S3)
│   └── test_compare_and_deploy.py # Deployment logic tests (mocked S3)
├── .github/workflows/
│   └── ci-cd.yml                 # Automated test → build → deploy pipeline
├── data/
│   └── creditcard_ci.csv         # 10k-row test dataset (committed to repo)
├── Dockerfile-training           # Training container (not deployed to AWS)
├── Dockerfile-inference          # API container (deployed to ECS)
└── requirements.txt
```

## Workflow

### Production Deployment Flow
1. **Train locally:** `python -m src.train_pipeline` (saves `-new` artifacts)
2. **Deploy if better:** `export API_URL=http://<ecs-ip>:8000 && python -m src.compare_and_deploy`
   - Compares F1 scores
   - If new > old: uploads to S3 + calls API `/reload`
   - If new ≤ old: keeps existing model, deletes `-new` artifacts
3. **API auto-reloads:** Downloads updated artifacts from S3, serves predictions with new model

### CI/CD Flow
- **Pull Request:** Integration tests run (fast feedback, no deployment)
- **Merge to main:** Tests → Build Docker images → Push to ECR → Update ECS task definition → Deploy new API version

## Running Locally

### Training & Deployment
```bash
# Train model (creates -new artifacts)
python -m src.train_pipeline

# Compare and conditionally deploy to S3
export API_URL=http://<ecs-public-ip>:8000
python -m src.compare_and_deploy
```

### Inference API (Local Testing)
```bash
uvicorn api.app:app --reload
# Visit http://localhost:8000/docs for interactive API documentation
```

### Testing
```bash
# Run all integration tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_training.py -v
```

## AWS Deployment

### Infrastructure
- **ECR:** `fraud-detection-api`, `fraud-detection-training` (Docker image registry)
- **S3:** `fraud-mlops-artifacts-bt/artifacts/` (model artifact storage)
- **ECS:** Fargate cluster with API service (0.25 vCPU, 0.5 GB memory)
- **IAM:** Task role with S3 read/write + ECS execution permissions

### API Endpoints
- **Health check:** `GET http://<ecs-public-ip>:8000/`
- **Prediction:** `POST http://<ecs-public-ip>:8000/predict`
- **Model reload:** `POST http://<ecs-public-ip>:8000/reload` (downloads from S3, returns F1 score)

### Example Prediction Request
```bash
curl -X POST http://<ecs-ip>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.0, 1.19, 0.27, 0.17, 0.45, 0.06, -0.08, -0.08, 0.09, -0.26, 
                 -0.17, 1.61, 1.07, 0.49, -0.14, 0.64, 0.46, -0.11, -0.18, -0.15,
                 -0.07, -0.23, -0.64, 0.10, -0.34, 0.17, 0.13, -0.01, 0.01, 2.69]
  }'

# Response:
# {"prediction": "legit", "fraud_probability": 0.0}
```

### Manual S3 Upload (if needed)
```bash
# Upload all artifacts to S3
aws s3 cp artifacts/model.joblib s3://fraud-mlops-artifacts-bt/artifacts/
aws s3 cp artifacts/preprocessor.joblib s3://fraud-mlops-artifacts-bt/artifacts/
aws s3 cp artifacts/threshold.joblib s3://fraud-mlops-artifacts-bt/artifacts/
aws s3 cp artifacts/best_f1.joblib s3://fraud-mlops-artifacts-bt/artifacts/
```

## Tech Stack
- **ML:** scikit-learn, pandas, numpy, joblib
- **API:** FastAPI, uvicorn, boto3 (AWS SDK)
- **Infrastructure:** Docker, AWS (ECS Fargate, ECR, S3, IAM)
- **CI/CD:** GitHub Actions, pytest, unittest.mock
- **Deployment:** Automated via GitHub Actions with conditional model promotion