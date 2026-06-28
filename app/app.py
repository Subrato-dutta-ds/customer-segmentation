import streamlit as st
import os

# --- IMPORT AUTO-TRAIN ---
from app.utils import ensure_model_exists
ensure_model_exists()
# -----------------------

st.set_page_config(
    page_title="Segmentation Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main > div { padding-top: 0rem; padding-bottom: 0rem; }
    .glass-card {
        background: rgba(26, 26, 46, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(108, 99, 255, 0.3);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
        margin-bottom: 1rem;
    }
    .feature-grid {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        margin-top: 20px;
    }
    .feature-card {
        background: rgba(26, 26, 46, 0.9);
        border-radius: 12px;
        padding: 20px 30px;
        border: 2px solid #2D323B;
        text-align: center;
        flex: 1;
        min-width: 120px;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .feature-card .number {
        font-size: 2.4rem;
        font-weight: 700;
        display: block;
    }
    .feature-card .label {
        font-size: 0.85rem;
        color: #AAAAAA;
    }
    .feature-card.purple { border-color: #6C63FF; }
    .feature-card.purple .number { color: #6C63FF; }
    .feature-card.pink { border-color: #FF6584; }
    .feature-card.pink .number { color: #FF6584; }
    .feature-card.cyan { border-color: #00D4FF; }
    .feature-card.cyan .number { color: #00D4FF; }
    .feature-card.green { border-color: #00E676; }
    .feature-card.green .number { color: #00E676; }
    .badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 4px;
    }
    .badge-purple { background: #6C63FF; color: white; }
    .badge-pink { background: #FF6584; color: white; }
    .badge-cyan { background: #00D4FF; color: #0F0F1A; }
    .badge-green { background: #00E676; color: #0F0F1A; }
    .divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #6C63FF, #FF6584, #00D4FF);
        margin: 1.5rem 0;
    }
    [data-testid="stSidebar"] {
        background: #0F0F1A;
        border-right: 1px solid rgba(108, 99, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="text-align: center; padding: 0.5rem 0;">
    <h1 style="font-size: 3.2rem; margin-bottom: 0; background: linear-gradient(135deg, #6C63FF, #FF6584, #00D4FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🛍️ Customer Segmentation Suite
    </h1>
    <p style="color: #AAAAAA; font-size: 1.2rem; margin-top: 0;">
        Real-time K-Means Clustering Engine
    </p>
    <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-top: 5px;">
        <span class="badge badge-purple">✅ Silhouette Score: 0.53</span>
        <span class="badge badge-pink">🔥 5 Segments</span>
        <span class="badge badge-cyan">📊 3 Features</span>
        <span class="badge badge-green">⚡ Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: rgba(26, 26, 46, 0.8); padding: 15px; border-radius: 12px; border: 1px solid rgba(108, 99, 255, 0.2); text-align: center;">
    <p style="color: #888; font-size: 0.8rem; margin: 0;">
        🚀 <b style="color: #6C63FF;">Placement Ready</b><br>
        <span style="font-size: 0.7rem; color: #666;">Built with Python, Streamlit, Scikit-learn</span>
    </p>
</div>
""", unsafe_allow_html=True)

# --- HOME PAGE CONTENT ---
st.markdown("""
<div class="glass-card">
    <h3 style="margin-top: 0;">🎯 Welcome to the Segmentation Engine</h3>
    <p style="color: #BBBBBB;">This application segments customers into <b style="color: #6C63FF;">5 distinct personas</b> using 
    <b style="color: #00D4FF;">K-Means Clustering</b> based on Income, Spending Score, and Age.</p>
    
    <div class="feature-grid">
        <div class="feature-card purple">
            <span class="number">0.53</span>
            <span class="label">📊 Silhouette Score</span>
        </div>
        <div class="feature-card pink">
            <span class="number">5</span>
            <span class="label">📌 Clusters</span>
        </div>
        <div class="feature-card cyan">
            <span class="number">3</span>
            <span class="label">📈 Features</span>
        </div>
        <div class="feature-card green">
            <span class="number">⚡</span>
            <span class="label">Live Streaming</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Model Status ---
if os.path.exists('models/kmeans_model.pkl'):
    st.success("✅ **Model Loaded** | Ready for predictions!")
else:
    st.info("⏳ **Model Training** | First load may take a few seconds...")