"""
DBSCAN vs K-Means Comparison Test
Run with: python test_dbscan.py
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.data_loader import load_data, clean_data

print('='*60)
print('🔬 DBSCAN vs K-MEANS COMPARISON')
print('='*60)

# Load data
print('\n📥 Loading data...')
df, sim = load_data(source='simulated', num_customers=300)
df = clean_data(df)

# Use 2 features (Income + Spending) for fair comparison
X = df[['Annual_Income_k', 'Spending_Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'\n📊 Dataset shape: {X_scaled.shape}')

# ============================================
# TEST 1: K-Means (Baseline)
# ============================================
print('\n' + '-'*60)
print('TEST 1: K-Means (Baseline)')
print('-'*60)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
print(f'✅ Silhouette Score: {kmeans_sil:.4f}')
print(f'📊 Clusters found: 5')

# ============================================
# TEST 2: DBSCAN (Try different parameters)
# ============================================
print('\n' + '-'*60)
print('TEST 2: DBSCAN (Grid Search)')
print('-'*60)

best_sil = -1
best_params = {}
results = []

# Try different eps and min_samples combinations
for eps in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
    for min_samples in [3, 5, 7, 10]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Only calculate silhouette if we have more than 1 cluster
        if n_clusters > 1:
            sil = silhouette_score(X_scaled, labels)
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'clusters': n_clusters,
                'noise': n_noise,
                'silhouette': sil
            })
            
            if sil > best_sil:
                best_sil = sil
                best_params = {'eps': eps, 'min_samples': min_samples, 'clusters': n_clusters, 'noise': n_noise}
            
            print(f'  eps={eps}, min_samples={min_samples} → {n_clusters} clusters, noise={n_noise}, Silhouette={sil:.4f}')
        else:
            print(f'  eps={eps}, min_samples={min_samples} → Only {n_clusters} cluster found (skipping)')

# ============================================
# RESULTS SUMMARY
# ============================================
print('\n' + '='*60)
print('📊 FINAL SUMMARY')
print('='*60)

print(f'\n🔵 K-Means (k=5):        Silhouette = {kmeans_sil:.4f}')
print(f'🟢 DBSCAN (Best):        Silhouette = {best_sil:.4f}')

if best_sil > kmeans_sil:
    print(f'\n✅ DBSCAN IS BETTER! Improvement of {((best_sil - kmeans_sil)/kmeans_sil)*100:.1f}%')
    print(f'\n🏆 Best DBSCAN Parameters:')
    print(f'   • eps = {best_params["eps"]}')
    print(f'   • min_samples = {best_params["min_samples"]}')
    print(f'   • Clusters found = {best_params["clusters"]}')
    print(f'   • Noise points = {best_params["noise"]}')
    print('\n💡 RECOMMENDATION: Use DBSCAN for your project!')
else:
    print(f'\n❌ K-Means is better or equal to DBSCAN on this dataset.')
    print(f'   Difference: {((kmeans_sil - best_sil)/kmeans_sil)*100:.1f}% worse')
    print('\n💡 RECOMMENDATION: Stick with K-Means, but mention you tested DBSCAN.')
    print('   This shows you understand advanced clustering methods!')

print('\n' + '='*60)
print('📌 INTERVIEW TIP: Mention that you experimented with both algorithms')
print('   and selected the one that gave the most actionable business segments.')
print('='*60)