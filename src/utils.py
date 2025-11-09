import joblib
import os


def save_artifacts(model, preprocessor, threshold, best_f1, output_dir='artifacts', suffix=''):
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = f"model{suffix}.joblib"
    preprocessor_name = f"preprocessor{suffix}.joblib"
    threshold_name = f"threshold{suffix}.joblib"
    best_f1_name = f"best_f1{suffix}.joblib"
    
    joblib.dump(model, os.path.join(output_dir, model_name))
    joblib.dump(preprocessor, os.path.join(output_dir, preprocessor_name))
    joblib.dump(threshold, os.path.join(output_dir, threshold_name))
    joblib.dump(best_f1, os.path.join(output_dir, best_f1_name))


def load_artifacts(input_dir='artifacts', suffix=''):
    
    model_name = f"model{suffix}.joblib"
    preprocessor_name = f"preprocessor{suffix}.joblib"
    threshold_name = f"threshold{suffix}.joblib"
    best_f1_name = f"best_f1{suffix}.joblib"
    
    model = joblib.load(os.path.join(input_dir, model_name))
    preprocessor = joblib.load(os.path.join(input_dir, preprocessor_name))
    threshold = joblib.load(os.path.join(input_dir, threshold_name))
    best_f1 = joblib.load(os.path.join(input_dir, best_f1_name))

    return model, preprocessor, threshold, best_f1