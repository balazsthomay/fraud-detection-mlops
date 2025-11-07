import pandas as pd
from typing import Tuple

def load_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load fraud detection dataset and split into features and target.
    
    Args:
        filepath: Path to CSV file containing fraud data
        
    Returns:
        Tuple of (X, y) where X is features dataframe and y is target series
        
    Raises:
        ValueError: If Class column is missing or dataframe is empty
    """
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if 'Class' not in df.columns:
        raise ValueError("Dataset missing 'Class' column")
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    return X, y