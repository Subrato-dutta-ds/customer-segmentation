"""
GMM vs K-Means Comparison Test
Run with: python test_gmm.py
"""

import sys
import os
sys.path.append('.')

import numpy as np
from src.data_loader import load_data, clean_data
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print('='*60)
print('🔬 K-MEANS vs GMM COMPARISON')
print('='*60)

# Load data
print('\n📥 Loading data...')
df, sim = load_data(source='simulated', num_customers=300)
df = clean_data(df)

# Use 3 features (Income, Spending, Age)
X = df[['Annual_Income_k', 'Spending_Score', 'Age']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'\n📊 Dataset shape: {X_scaled.shape}')
print(f'📊 Features: Income, Spending, Age')

# ============================================
# TEST 1: K-Means (Baseline)
# ============================================
print('\n' + '-'*60)
print('TEST 1: K-Means (Baseline)')
print('-'*60)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
print(f'✅ Silhouette Score: {kmeans_sil:.4f}')

# ============================================
# TEST 2: GMM (Different covariance types)
# ============================================
print('\n' + '-'*60)
print('TEST 2: Gaussian Mixture Models')
print('-'*60)

covariance_types = ['full', 'tied', 'diag', 'spherical']
best_gmm_sil = -1
best_gmm_params = {}

for cov_type in covariance_types:
    gmm = GaussianMixture(
        n_components=5,
        covariance_type=cov_type,
        random_state=42,
        max_iter=200,
        init_params='kmeans'
    )
    gmm_labels = gmm.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, gmm_labels)
    print(f'  {cov_type}: Silhouette = {sil:.4f}')
    
    if sil > best_gmm_sil:
        best_gmm_sil = sil
        best_gmm_params = {'covariance_type': cov_type}

# ============================================
# TEST 3: GMM with different number of components
# ============================================
print('\n' + '-'*60)
print('TEST 3: GMM - Finding Optimal Components')
print('-'*60)

gmm_scores = []
for n_components in range(2, 8):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=best_gmm_params['covariance_type'],
        random_state=42,
        max_iter=200
    )
    labels = gmm.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    gmm_scores.append(sil)
    print(f'  n_components={n_components}: Silhouette = {sil:.4f}')

best_n = range(2, 8)[np.argmax(gmm_scores)]
print(f'\n✅ Best n_components: {best_n} (score: {max(gmm_scores):.4f})')

# ============================================
# RESULTS SUMMARY
# ============================================
print('\n' + '='*60)
print('📊 FINAL SUMMARY')
print('='*60)

print(f'\n🔵 K-Means (k=5):        Silhouette = {kmeans_sil:.4f}')
print(f'🟢 GMM (Best):           Silhouette = {best_gmm_sil:.4f}')

if best_gmm_sil > kmeans_sil:
    print(f'\n✅ GMM IS BETTER! Improvement of {((best_gmm_sil - kmeans_sil)/kmeans_sil)*100:.1f}%')
    print(f'\n🏆 Best GMM Parameters:')
    print(f'   • Covariance Type: {best_gmm_params["covariance_type"]}')
    print(f'   • Optimal Components: {best_n}')
    print(f'   • Silhouette Score: {best_gmm_sil:.4f}')
    print('\n💡 RECOMMENDATION: Use GMM for your project!')
    print('   GMM handles overlapping customer segments better than K-Means.')
    print('   It also provides probability scores for each customer.')
else:
    print(f'\n❌ K-Means is still better or equal to GMM.')
    print(f'   Difference: {((kmeans_sil - best_gmm_sil)/kmeans_sil)*100:.1f}%')
    print('\n💡 RECOMMENDATION: Stick with K-Means, but mention you tested GMM.')

print('\n' + '='*60)
print('📌 INTERVIEW TIP:')
print('   "I compared K-Means with GMM and selected the model that gave')
print('   the best balance of performance and business interpretability."')
print('='*60)