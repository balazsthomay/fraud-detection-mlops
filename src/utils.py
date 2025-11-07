import joblib
import os


def save_artifacts(model, preprocessor, threshold, output_dir='artifacts'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    joblib.dump(threshold, os.path.join(output_dir, 'threshold.joblib'))


def load_artifacts(input_dir='artifacts'):
    
    model = joblib.load(os.path.join(input_dir, 'model.joblib'))
    preprocessor = joblib.load(os.path.join(input_dir, 'preprocessor.joblib'))
    threshold = joblib.load(os.path.join(input_dir, 'threshold.joblib'))

    return model, preprocessor, threshold