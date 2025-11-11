# servers predictions

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import boto3
from src.utils import load_artifacts

app = FastAPI()

S3_BUCKET = os.getenv('S3_BUCKET', 'fraud-mlops-artifacts-bt') # which S3 bucket
S3_PREFIX = 'artifacts/'  # folder in S3 bucket

# artifacts_path = os.getenv('ARTIFACTS_PATH', 'artifacts') # this line is redundant, used for local

try:
    # download from S3 and load artifacts on startup
    s3_client = boto3.client('s3')
    for artifact in ["model", "preprocessor", "threshold", "best_f1"]:
        s3_client.download_file(
            Bucket=S3_BUCKET,
            Key=f'{S3_PREFIX}{artifact}.joblib',
            Filename=f'/tmp/{artifact}.joblib'
        )
    model, preprocessor, threshold, best_f1 = load_artifacts('/tmp')
    print("Models loaded from S3 successfully")
except Exception as e:
    print(f"Could not load models on startup: {e}")
    # Set to None so API knows models aren't ready
    model = preprocessor = threshold = best_f1 = None

class Transaction(BaseModel):
    features: List[float]  # Expects a list of 30 floats


@app.get("/")
def read_root():
    return {"message": "Fraud Detection API"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Check if models are loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /reload endpoint first.")
    
    # Validate input length
    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features")
    
    # Convert to numpy array and reshape for preprocessing
    X = np.array(transaction.features).reshape(1, -1)
    
    # Preprocess (scale the features)
    X_scaled = preprocessor.transform(X)
    
    # Get prediction probability
    y_proba = model.predict_proba(X_scaled)[0][1]  # Probability of fraud
    
    # Apply threshold
    y_pred = 1 if y_proba >= threshold else 0
    
    return {
        "prediction": "fraud" if y_pred == 1 else "legit",
        "fraud_probability": float(y_proba)
    }
    
@app.post("/reload")
def reload_model():
    global model, preprocessor, threshold, best_f1
    
    # Download artifacts from S3 to local temp directory
    s3_client = boto3.client('s3')
    for artifact in ["model", "preprocessor", "threshold", "best_f1"]:
        s3_client.download_file(
            Bucket=S3_BUCKET,
            Key=f'{S3_PREFIX}{artifact}.joblib',  # Path in S3
            Filename=f'/tmp/{artifact}.joblib'    # Where to save in container
        )
    
    # Load artifacts from temp directory
    model, preprocessor, threshold, best_f1 = load_artifacts(input_dir='/tmp')
    
    # Return success message
    return {
        "status": "model reloaded successfully",
        "model_f1_score": float(best_f1)
    }



# full flow WITH AWS S3: train locally → upload to S3 → call API reload → verify predictions work
