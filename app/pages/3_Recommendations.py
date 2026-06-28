import streamlit as st
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Recommendations", layout="wide")
st.title("💡 Marketing Playbook")

@st.cache_resource
def load_profiles():
    try:
        return pd.DataFrame(joblib.load('models/cluster_profiles.pkl'))
    except:
        return None

profiles = load_profiles()

if profiles is None:
    st.error("❌ Run `python src/train.py` first.")
    st.stop()

st.markdown("---")

# Gradient Color Palette
colors = ['#6C63FF', '#FF6584', '#00D4FF', '#00E676', '#FFB74D']
icons = ['💎', '📈', '🛒', '🔥', '👴']
names = ['Premium VIP', 'Standard', 'Budget', 'Impulse Spender', 'Retired']
gradients = [
    'linear-gradient(135deg, #6C63FF, #4A3FBF)',
    'linear-gradient(135deg, #FF6584, #CC3366)',
    'linear-gradient(135deg, #00D4FF, #0099CC)',
    'linear-gradient(135deg, #00E676, #00A855)',
    'linear-gradient(135deg, #FFB74D, #FF9800)'
]

def get_strategy(cluster):
    strategies = {
        0: "Exclusive perks, VIP events, and 15% loyalty cashback.",
        1: "Personalized upselling, cross-selling, and product bundles.",
        2: "Value deals, bulk discounts, and budget-friendly packages.",
        3: "Social media campaigns, flash sales, and referral bonuses.",
        4: "Trust campaigns, easy-to-use interfaces, and senior discounts."
    }
    return strategies.get(cluster, "Standard retention marketing.")

# --- Display each segment ---
for idx, row in profiles.iterrows():
    cluster = int(row['Cluster'])
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"""
        <div style="background: {gradients[cluster]}; border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <h1 style="font-size: 3rem; margin: 0;">{icons[cluster]}</h1>
            <h2 style="margin: 5px 0; color: white;">Segment {cluster}</h2>
            <p style="margin: 0; color: rgba(255,255,255,0.9); font-weight: bold;">{names[cluster]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        size_text = ""
        if 'Size' in profiles.columns:
            size_text = f" | 👥 Size: {int(row['Size'])} customers"
        
        st.markdown(f"""
        <div style="background: rgba(26, 26, 46, 0.9); border-radius: 16px; padding: 20px; height: 100%; border: 1px solid {colors[cluster]}33;">
            <b style="color: {colors[cluster]};">📋 Profile:</b><br>
            💰 Income: {row['Annual_Income_k']:.1f}k | 
            ⭐ Spending: {row['Spending_Score']:.1f}
            {size_text}
            <br><br>
            <b style="color: {colors[cluster]};">🚀 Recommended Strategy:</b><br>
            <span style="color: #DDDDDD;">{get_strategy(cluster)}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")