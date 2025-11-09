# train_pipeline.py → Trains model, saves artifacts, exits
# compare_and_deploy.py → Loads old & new models, compares F1, calls /reload if better

# Production model is model.joblib
# Training saves new model as model-new.joblib
# Load both
# Compare both
# If new is better: delete model.joblib, rename model-new.joblib → model.joblib, call /reload
# If new is worse: delete model-new.joblib, keep model.joblib

from src.utils import load_artifacts
import os
import requests

artifacts_path = os.getenv('ARTIFACTS_PATH', 'artifacts')
api_url = os.getenv('API_URL')

def main():
    
    if os.path.exists(os.path.join(artifacts_path, f"model-new.joblib")):    # if articles exist
        
        # load old and new models
        model, preprocessor, threshold, best_f1 = load_artifacts(artifacts_path)
        model_new, preprocessor_new, threshold_new, best_f1_new = load_artifacts(artifacts_path, suffix='-new')
        
        if best_f1_new > best_f1:  # new model is better, replace old with new
            
            for artifact in ["model", "preprocessor", "threshold", "best_f1"]:
                os.remove(os.path.join(artifacts_path, f"{artifact}.joblib"))
                os.rename(os.path.join(artifacts_path, f"{artifact}-new.joblib"), os.path.join(artifacts_path, f"{artifact}.joblib"))
            print(f"New model F1: {best_f1_new} is better than old model F1: {best_f1}. Deploying new model.")
            
            # notify API to reload model
            if api_url:
                try:
                    response = requests.post(f"{api_url}/reload") # Notify inference API to reload model only in Docker mode
                    print(f"Model reload triggered: {response.json()}")
                except Exception as e:
                    print(f"Could not notify API: {e}")  
        
        else:
            for artifact in ["model-new", "preprocessor-new", "threshold-new", "best_f1-new"]:
                os.remove(os.path.join(artifacts_path, f"{artifact}.joblib"))
            print(f"Old model F1: {best_f1} is better than new model F1: {best_f1_new}. Keeping old model.")
            
    else:     # if articles don't exist, exit gracefully
        print("Artifacts couldn't be loaded")
    

if __name__ == "__main__":
    main()