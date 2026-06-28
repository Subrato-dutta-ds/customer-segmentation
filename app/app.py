import streamlit as st
import os

st.set_page_config(
    page_title="Segmentation Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for minor tweaks only ---
st.markdown("""
<style>
    /* Only minimal styling */
    .main > div { padding-top: 0rem; padding-bottom: 0rem; }
    [data-testid="stSidebar"] { background: #0F0F1A; }
    
    /* Gradient text for headers */
    .gradient-text {
        background: linear-gradient(135deg, #6C63FF, #FF6584, #00D4FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Badge style */
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
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1rem 0;">
    <h1 style="font-size: 3rem; margin-bottom: 0; background: linear-gradient(135deg, #6C63FF, #FF6584, #00D4FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🛍️ Customer Segmentation Suite
    </h1>
    <p style="color: #AAAAAA; font-size: 1.1rem; margin-top: 0;">
        Real-time K-Means Clustering Engine
    </p>
    <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-top: 5px;">
        <span class="badge badge-purple">✅ Score: 0.53</span>
        <span class="badge badge-pink">🔥 5 Segments</span>
        <span class="badge badge-cyan">📊 3 Features</span>
        <span class="badge badge-green">⚡ Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- SIDEBAR FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: rgba(26, 26, 46, 0.8); padding: 15px; border-radius: 12px; border: 1px solid rgba(108, 99, 255, 0.2); text-align: center;">
    <p style="color: #888; font-size: 0.8rem; margin: 0;">
        🚀 <b style="color: #6C63FF;">Placement Ready</b><br>
        <span style="font-size: 0.7rem; color: #666;">Built with Python, Streamlit, Scikit-learn</span>
    </p>
</div>
""", unsafe_allow_html=True)

# --- MAIN CONTENT: Using Streamlit Native Components ---

# Welcome Section
st.markdown("""
### 🎯 Welcome to the Segmentation Engine

This application segments customers into **5 distinct personas** using 
**K-Means Clustering** based on Income, Spending Score, and Age.
""")

st.markdown("---")

# --- FEATURE CARDS USING STREAMLIT COLUMNS (Guaranteed to work!) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: rgba(108, 99, 255, 0.15); border: 2px solid #6C63FF; border-radius: 12px; padding: 20px; text-align: center;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #6C63FF;">0.53</div>
        <div style="color: #AAAAAA; font-size: 0.9rem;">📊 Silhouette Score</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: rgba(255, 101, 132, 0.15); border: 2px solid #FF6584; border-radius: 12px; padding: 20px; text-align: center;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #FF6584;">5</div>
        <div style="color: #AAAAAA; font-size: 0.9rem;">📌 Clusters</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.15); border: 2px solid #00D4FF; border-radius: 12px; padding: 20px; text-align: center;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #00D4FF;">3</div>
        <div style="color: #AAAAAA; font-size: 0.9rem;">📈 Features</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: rgba(0, 230, 118, 0.15); border: 2px solid #00E676; border-radius: 12px; padding: 20px; text-align: center;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #00E676;">⚡</div>
        <div style="color: #AAAAAA; font-size: 0.9rem;">Live Streaming</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Model Status ---
if os.path.exists('models/kmeans_model.pkl'):
    st.success("✅ **Model Loaded** | Ready for predictions!")
else:
    st.error("❌ **Model Missing** | Please run `python src/train.py` first.")