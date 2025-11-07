from src.data import load_data
from src.preprocessing import FraudPreprocessor, split_data
from src.models import train_random_forest
from src.evaluate import find_optimal_threshold, evaluate_model
from src.utils import save_artifacts

def main(filepath, test_size=0.2, random_state=42, n_estimators=100, ):
    
    X, y = load_data(filepath) # load data, get features/target
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state) # split dataset

    fraud_processor = FraudPreprocessor()
    X_train_scaled = fraud_processor.fit_transform(X_train) # standardize features
    X_test_scaled = fraud_processor.transform(X_test)

    model = train_random_forest(X_train_scaled, y_train, n_estimators=n_estimators, random_state=42) # train model
    
    y_pred = model.predict(X_test_scaled) # Get class predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Get fraud probabilities
    
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_pred_proba, n_thresholds=100)
    evaluation = evaluate_model(y_test, y_pred)
    print(f"Best threshold: {best_threshold} \n Best F1 score: {best_f1}")
    print(f"Evaluation report: {evaluation}")
    
    save_artifacts(model, fraud_processor, best_threshold, output_dir='../artifacts')


if __name__ == "__main__":
    main(filepath='./data/creditcard.csv')