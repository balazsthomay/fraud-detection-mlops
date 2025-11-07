from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np


def find_optimal_threshold(y_true, y_pred_proba, n_thresholds=100):
    """
    Returns best_threshold, best_f1
    """
        
    threshold_samples = np.linspace(0, 1, n_thresholds) # Sample thresholds (e.g., 100 evenly spaced from 0 to 1)

    f1_scores = []
    for threshold in threshold_samples:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_threshold)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = threshold_samples[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1


def evaluate_model(y_true, y_pred):
    """
    Returns dict with precision, recall, f1, and classification report
    """

    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred)
    }