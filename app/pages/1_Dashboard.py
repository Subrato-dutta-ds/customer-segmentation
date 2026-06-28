import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import sys
import os

# Auto-train if model missing
from app.utils import ensure_model_exists
ensure_model_exists()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 Dashboard Overview")

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/kmeans_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_artifacts()

if model is None:
    st.error("❌ Model not found. Run `python src/train.py`")
    st.stop()

@st.cache_data
def get_data():
    from src.data_loader import load_data, clean_data
    df, _ = load_data(source='simulated', num_customers=300)
    df = clean_data(df)
    X = df[['Annual_Income_k', 'Spending_Score', 'Age']]
    X_scaled = scaler.transform(X)
    df['Cluster'] = model.predict(X_scaled)
    return df

df = get_data()

# --- Gradient Color Palette for Segments ---
segment_colors = ['#6C63FF', '#FF6584', '#00D4FF', '#00E676', '#FFB74D']

# --- METRIC ROW (INR Currency) ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("👥 Total Customers", len(df), delta="300")
col2.metric("📌 Clusters", 5, delta="K-Means")
col3.metric("📈 Silhouette", "0.53", delta="Good")
col4.metric("💰 Avg Income", f"₹{df['Annual_Income_k'].mean():.1f}k")
col5.metric("⭐ Avg Spending", f"{df['Spending_Score'].mean():.1f}")

st.markdown("---")

# --- ROW 1: Scatter Plot + Distribution ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧩 Segment Distribution")
    fig = px.scatter(
        df,
        x='Annual_Income_k',
        y='Spending_Score',
        color='Cluster',
        color_discrete_sequence=segment_colors,
        hover_data=['Age'],
        title='Income vs Spending (Colored by Cluster)',
        labels={'Annual_Income_k': 'Annual Income (₹k)', 'Spending_Score': 'Spending Score (1-100)'}
    )
    fig.update_layout(height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Segment Sizes")
    size_df = df['Cluster'].value_counts().sort_index().reset_index()
    size_df.columns = ['Cluster', 'Count']
    fig2 = px.bar(
        size_df,
        x='Cluster',
        y='Count',
        text='Count',
        color='Cluster',
        color_discrete_sequence=segment_colors,
        title='Customers per Segment'
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --- ROW 2: Cluster Profiles (INR Currency) ---
st.subheader("📋 Segment Profiles")

profile_df = df.groupby('Cluster').agg({
    'Annual_Income_k': 'mean',
    'Spending_Score': 'mean',
    'Age': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Size'}).reset_index()

display_df = profile_df.copy()
display_df['Annual_Income_k'] = display_df['Annual_Income_k'].map('₹{:.1f}k'.format)
display_df['Spending_Score'] = display_df['Spending_Score'].map('{:.1f}'.format)
display_df['Age'] = display_df['Age'].map('{:.1f} yrs'.format)
display_df['Size'] = display_df['Size'].map('{:.0f}'.format)

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")

# --- ROW 3: CLUSTER VISUALIZATION PNG ---
st.subheader("🖼️ Cluster Visualization")

if os.path.exists('models/cluster_visualization.png'):
    st.image('models/cluster_visualization.png', caption='Customer Segments Visualization', use_container_width=True)
else:
    st.info("ℹ️ Visualization not available. Please run `python src/train.py` first.")

# --- INSIGHTS ---
with st.expander("💡 Business Insights (Click to expand)"):
    st.markdown("""
    | Cluster | Persona | Strategy |
    |---------|---------|----------|
    | 0 🟣 | **Premium VIP** | High income, high spend. Target with VIP perks. |
    | 1 🔴 | **Standard** | Medium income, medium spend. Focus on upselling. |
    | 2 🔵 | **Budget** | Low income, low spend. Value deals and bulk offers. |
    | 3 🟢 | **Impulse Spender** | Low income, high spend. Social media & flash sales. |
    | 4 🟠 | **Retired** | Medium-low income, moderate spend. Trust-building campaigns. |
    """)