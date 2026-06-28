import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to Python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Now import from src using absolute import
from src.data_loader import load_data, clean_data
from src.preprocess import select_features, scale_features, save_scaler

def find_optimal_k(X_scaled, max_k=10):
    """Calculate inertias and silhouette scores for k=2 to max_k."""
    inertias = []
    silhouette_scores = []
    
    print("🔍 Testing different K values...")
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
    
    return inertias, silhouette_scores

def train_kmeans(X_scaled, k=5):
    """Train K-Means with the specified number of clusters."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    print(f"✅ K-Means trained with k={k}")
    return kmeans

def save_model(model, path='models/kmeans_model.pkl'):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def main():
    print("="*60)
    print("🏋️ TRAINING CUSTOMER SEGMENTATION MODEL")
    print("="*60)
    
    # Load data (use simulated data for training)
    print("\n📥 Loading data...")
    df, sim = load_data(source='simulated', num_customers=400)
    df = clean_data(df)
    
    # Preprocess
    print("\n🔧 Preprocessing...")
    X = select_features(df)
    X_scaled, scaler = scale_features(X)
    
    # Find optimal K
    print("\n" + "-"*40)
    inertias, silhouette_scores = find_optimal_k(X_scaled, max_k=10)
    
    # Choose best k (highest silhouette score)
    best_k = range(2, 11)[np.argmax(silhouette_scores)]
    print("\n" + "-"*40)
    print(f"✅ Optimal k = {best_k} (highest silhouette score: {max(silhouette_scores):.4f})")
    
    # Train final model
    print(f"\n🎯 Training final model with k={best_k}...")
    model = train_kmeans(X_scaled, k=best_k)
    
    # Save everything
    save_model(model)
    save_scaler(scaler)
    
    # --- UPDATED: Save cluster profiles with Size ---
    df['Cluster'] = model.predict(X_scaled)
    profiles = df.groupby('Cluster').agg({
        'Annual_Income_k': 'mean',
        'Spending_Score': 'mean',
        'Age': 'mean'
    }).reset_index()
    # Add cluster size
    size_series = df.groupby('Cluster').size()
    profiles['Size'] = profiles['Cluster'].map(size_series)
    joblib.dump(profiles, 'models/cluster_profiles.pkl')
    print("✅ Cluster profiles saved to models/cluster_profiles.pkl")
    
    # Quick visualisation
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Annual_Income_k', y='Spending_Score', hue='Cluster', data=df, palette='viridis')
    plt.title(f'Customer Segments (k={best_k})')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title='Cluster')
    plt.savefig('models/cluster_visualization.png')
    plt.close()
    print("✅ Cluster visualization saved to models/cluster_visualization.png")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE! Model ready for live predictions.")
    print(f"📊 Optimal clusters: {best_k}")
    print("="*60)

if __name__ == "__main__":
    main()