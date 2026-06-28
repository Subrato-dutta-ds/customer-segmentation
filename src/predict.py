import joblib
import numpy as np
import pandas as pd

def load_models(model_path='models/kmeans_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Load the trained K-Means model and StandardScaler from disk.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Models loaded successfully!")
    return model, scaler

def predict_cluster(features, model, scaler):
    """
    Predict the cluster for a single customer or batch of customers.
    
    Parameters:
    - features: dict or DataFrame with 'Annual_Income_k' and 'Spending_Score'
    - model: trained KMeans model
    - scaler: fitted StandardScaler
    
    Returns:
    - cluster label(s)
    """
    if isinstance(features, dict):
        X = np.array([[features['Annual_Income_k'], features['Spending_Score']]])
    else:
        X = features[['Annual_Income_k', 'Spending_Score']].values
    
    X_scaled = scaler.transform(X)
    clusters = model.predict(X_scaled)
    return clusters

def predict_and_add_clusters(df, model, scaler):
    """
    Add cluster predictions to an existing DataFrame.
    """
    if 'Annual_Income_k' in df.columns and 'Spending_Score' in df.columns:
        X = df[['Annual_Income_k', 'Spending_Score']]
        X_scaled = scaler.transform(X)
        df['Cluster'] = model.predict(X_scaled)
    return df

# Test block (runs only when you execute this file directly)
if __name__ == "__main__":
    print("Testing predict.py...")
    model, scaler = load_models()
    test_features = {'Annual_Income_k': 60, 'Spending_Score': 50}
    cluster = predict_cluster(test_features, model, scaler)
    print(f"Test customer (Income=60, Spending=50) → Cluster {cluster[0]}")