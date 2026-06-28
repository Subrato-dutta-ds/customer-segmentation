import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def select_features(df, feature_cols=['Annual_Income_k', 'Spending_Score', 'Age']):
    """
    Extract the features we want to use for clustering.
    """
    X = df[feature_cols]
    print(f"✅ Selected features: {feature_cols}")
    return X

def scale_features(X):
    """
    Scale the features using StandardScaler (mean=0, std=1).
    Returns: scaled array, scaler object.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✅ Features scaled successfully. Shape: {X_scaled.shape}")
    return X_scaled, scaler

def save_scaler(scaler, path='models/scaler.pkl'):
    """
    Save the fitted scaler to disk for later use in prediction.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"✅ Scaler saved to {path}")

# Test block
if __name__ == "__main__":
    # Test with dummy data
    import sys
    sys.path.append('.')
    from data_loader import load_data, clean_data
    
    df, sim = load_data(source='simulated', num_customers=30)
    df = clean_data(df)
    X = select_features(df)
    X_scaled, scaler = scale_features(X)
    save_scaler(scaler)
    print("\n✅ Preprocessing test successful!")
    print(f"Sample scaled data (first 3 rows):\n{X_scaled[:3]}")