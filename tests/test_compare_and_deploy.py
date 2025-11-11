import joblib
import os
from src.compare_and_deploy import main as compare_and_deploy
from src.utils import load_artifacts
import shutil
import pytest
from unittest.mock import patch, MagicMock


artifacts_path = os.getenv('ARTIFACTS_PATH', 'artifacts')

@pytest.fixture
def backup_baseline_artifacts():
    """Backup baseline artifacts before test, restore after"""
    # Setup: Create backups
    shutil.copy('artifacts/model.joblib', 'artifacts/model.backup')
    shutil.copy('artifacts/preprocessor.joblib', 'artifacts/preprocessor.backup')
    shutil.copy('artifacts/threshold.joblib', 'artifacts/threshold.backup')
    shutil.copy('artifacts/best_f1.joblib', 'artifacts/best_f1.backup')
    
    yield  # Test runs here
    
    # Teardown: Restore from backups
    shutil.copy('artifacts/model.backup', 'artifacts/model.joblib')
    shutil.copy('artifacts/preprocessor.backup', 'artifacts/preprocessor.joblib')
    shutil.copy('artifacts/threshold.backup', 'artifacts/threshold.joblib')
    shutil.copy('artifacts/best_f1.backup', 'artifacts/best_f1.joblib')
    
    # Cleanup: Remove backups
    os.remove('artifacts/model.backup')
    os.remove('artifacts/preprocessor.backup')
    os.remove('artifacts/threshold.backup')
    os.remove('artifacts/best_f1.backup')
    

@patch('src.compare_and_deploy.boto3.client')
def test_compare_and_deploy_better_model(mock_boto3, backup_baseline_artifacts):
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3.return_value = mock_s3
    
    # Copy baseline artifacts to create -new versions
    shutil.copy('artifacts/model.joblib', 'artifacts/model-new.joblib')
    shutil.copy('artifacts/preprocessor.joblib', 'artifacts/preprocessor-new.joblib')
    shutil.copy('artifacts/threshold.joblib', 'artifacts/threshold-new.joblib')
    
    # Override F1 score to make it better
    joblib.dump(0.95, 'artifacts/best_f1-new.joblib')
    
    compare_and_deploy()
    model, preprocessor, threshold, best_f1 = load_artifacts(artifacts_path)
    
    assert best_f1 == 0.95
    

@patch('src.compare_and_deploy.boto3.client')
def test_compare_and_deploy_worse_model(mock_boto3, backup_baseline_artifacts):
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3.return_value = mock_s3
    
    # Copy baseline artifacts to create -new versions
    shutil.copy('artifacts/model.joblib', 'artifacts/model-new.joblib')
    shutil.copy('artifacts/preprocessor.joblib', 'artifacts/preprocessor-new.joblib')
    shutil.copy('artifacts/threshold.joblib', 'artifacts/threshold-new.joblib')
    
    # Override F1 score to make it worse
    joblib.dump(0.70, 'artifacts/best_f1-new.joblib')
    
    compare_and_deploy()
    model, preprocessor, threshold, best_f1 = load_artifacts(artifacts_path)
    
    assert best_f1 != 0.70