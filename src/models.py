from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train, max_iter=100, random_state=42):
    
    model = LogisticRegression(
        class_weight='balanced', 
        random_state=random_state)
    model = model.fit(X_train, y_train)
    
    return model
    
    
def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=random_state,
        n_estimators=n_estimators
    )
    rf_model = rf_model.fit(X_train, y_train)
    
    return rf_model
    
    
def train_xgboost(X_train, y_train, n_estimators=100, random_state=42):
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_estimators=n_estimators
    )
    xgb_model = xgb_model.fit(X_train, y_train)
    
    return xgb_model