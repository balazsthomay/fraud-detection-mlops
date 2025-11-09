# servers predictions

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import os
from src.utils import load_artifacts

app = FastAPI()

artifacts_path = os.getenv('ARTIFACTS_PATH', 'artifacts')
model, preprocessor, threshold, best_f1 = load_artifacts(artifacts_path) # Load artifacts once at startup

# model, preprocessor, threshold = load_artifacts("../artifacts")

class Transaction(BaseModel):
    features: List[float]  # Expects a list of 30 floats


@app.get("/")
def read_root():
    return {"message": "Fraud Detection API"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
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
        "prediction": "fraud" if y_pred == 1 else "legitimate",
        "fraud_probability": float(y_proba)
    }
    
@app.post("/reload")
def reload_model():
    global model, preprocessor, threshold
    model, preprocessor, threshold, best_f1 = load_artifacts(artifacts_path)
    return {"status": "model reloaded successfully"}