import pandas as pd
from src.train_pipeline import main as train_pipeline
from src.utils import load_artifacts
import os


data_path = os.getenv('DATA_PATH', './data/creditcard_ci.csv')
artifacts_path = os.getenv('ARTIFACTS_PATH', 'artifacts')



def test_training_pipeline():
    # check if the training pipeline runs without errors on a small subset of data
    
    df_full = pd.read_csv(data_path)
    fraud = df_full[df_full['Class'] == 1].sample(n=50, random_state=42)  # Get 50 fraud cases
    legit = df_full[df_full['Class'] == 0].sample(n=950, random_state=42)  # Get 950 legit cases
    
    df_temp = pd.concat([fraud, legit]).sample(frac=1, random_state=42)  # Shuffle
    df_temp.to_csv('data/creditcard_temp.csv', index=False)
    temp_data_path = os.getenv('TEMP_DATA_PATH', './data/creditcard_temp.csv')
                
    train_pipeline(filepath=temp_data_path)
    
    for artifact in ["model", "preprocessor", "threshold", "best_f1"]:
        assert(os.path.exists(os.path.join(artifacts_path, f"{artifact}-new.joblib"))) # checking if files are created
       
    model, preprocessor, threshold, best_f1 = load_artifacts(suffix="-new")
    
    assert model is not None
    assert preprocessor is not None
    assert threshold is not None
    assert best_f1 is not None
    assert threshold > 0 and threshold < 1
    assert best_f1 > 0 and best_f1 <= 1
    
    sample_X = df_temp.drop('Class', axis=1).iloc[0:1]  # First row, keep as DataFrame
    sample_scaled = preprocessor.transform(sample_X)
    prediction = model.predict(sample_scaled)
    assert prediction is not None  # Verify prediction worked
    
    for artifact in ["model-new", "preprocessor-new", "threshold-new", "best_f1-new"]:
        os.remove(os.path.join(artifacts_path, f"{artifact}.joblib"))
        
    os.remove(temp_data_path)
    
# pytest tests/test_training.py -v