"""
Test script to compare model with and without Age feature
Run with: python test_with_age.py
"""

import sys
import os
sys.path.append('.')

from src.data_loader import load_data, clean_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print('='*50)
print('📊 TESTING WITH AGE INCLUDED')
print('='*50)

# Load data
print('\n📥 Loading data...')
df, sim = load_data(source='simulated', num_customers=300)
df = clean_data(df)

# Original (2 features)
print('\n🔵 Original (Income + Spending):')
X1 = df[['Annual_Income_k', 'Spending_Score']]
X1_scaled = StandardScaler().fit_transform(X1)
labels1 = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X1_scaled)
sil1 = silhouette_score(X1_scaled, labels1)
print(f'   Silhouette Score: {sil1:.4f}')

# With Age (3 features)
print('\n🟢 With Age (Income + Spending + Age):')
X2 = df[['Annual_Income_k', 'Spending_Score', 'Age']]
X2_scaled = StandardScaler().fit_transform(X2)
labels2 = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X2_scaled)
sil2 = silhouette_score(X2_scaled, labels2)
print(f'   Silhouette Score: {sil2:.4f}')

# Calculate improvement
improvement = ((sil2 - sil1) / sil1) * 100
print(f'\n📈 Improvement: {improvement:+.1f}%')

# Recommendation
print('\n' + '='*50)
if sil2 > sil1:
    print('✅ RECOMMENDATION: Retrain with Age included!')
    print('   Run: python src/train.py (after updating select_features)')
else:
    print('❌ Stick with 2 features.')
    print('   Age doesn\'t improve the model significantly.')
print('='*50)