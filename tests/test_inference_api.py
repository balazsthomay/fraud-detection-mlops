import os
from fastapi.testclient import TestClient
from api.app import app
import pandas as pd


data_path = os.getenv('DATA_PATH', './data/creditcard_ci.csv')


def test_inference_api():
    # verifies API can load models and serve predictions
    # 1. Start FastAPI server in background
    client = TestClient(app)    
    
    # 2. Send prediction request
    df = pd.read_csv(data_path)
    valid_transaction = df.drop('Class', axis=1).iloc[1].tolist()
    fraud_transaction = df[df['Class'] == 1].drop('Class', axis=1).iloc[0].tolist()    
    
    response_valid = client.post("/predict", json={"features": valid_transaction})
    response_fraud = client.post("/predict", json={"features": fraud_transaction})
    
    # 3. Verify response
    assert response_valid.status_code == 200
    assert response_valid.json()["prediction"] == "legit"
    assert 0 <= response_valid.json()["fraud_probability"] <= 1
    
    assert response_fraud.status_code == 200
    assert response_fraud.json()["prediction"] == "fraud"
    assert 0 <= response_fraud.json()["fraud_probability"] <= 1


# run all 3 tests on every PR