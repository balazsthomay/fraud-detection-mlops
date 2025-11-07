import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_fraud_probabilities(y_test, y_pred_proba):
    
    fraud_probs = y_pred_proba[y_test == 1]
    non_fraud_probs = y_pred_proba[y_test == 0]

    print(f"Fraud probability stats:")
    print(f"Min: {fraud_probs.min()}, Max: {fraud_probs.max()}, Mean: {fraud_probs.mean()}")
    print(f"\nNon-fraud probability stats:")
    print(f"Min: {non_fraud_probs.min()}, Max: {non_fraud_probs.max()}, Mean: {non_fraud_probs.mean()}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(fraud_probs, bins=50, alpha=0.7, color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Fraud Cases')

    plt.subplot(1, 2, 2)
    plt.hist(non_fraud_probs, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count (log scale)')
    plt.yscale('log')
    plt.title('Non-Fraud Cases')
    plt.tight_layout()
    plt.show()
    

def plot_precision_recall_vs_threshold(y_test, y_pred_proba):
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Plot precision vs recall at different thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision and Recall vs Threshold')
    plt.grid(True)
    plt.show()
    