from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FraudPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X_train):
        self.scaler.fit(X_train)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)
    
    
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test