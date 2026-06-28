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

# Color palette for segments
colors = ['#00CC96', '#FFB74D', '#FF6B6B', '#845EC2', '#00C9A7']
icons = ['💎', '📈', '🛒', '🔥', '👴']
names = ['Premium VIP', 'Standard', 'Budget', 'Impulse Spender', 'Retired']

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
        <div style="background: {colors[cluster]}20; border: 2px solid {colors[cluster]}; border-radius: 12px; padding: 15px; text-align: center;">
            <h1 style="font-size: 2.5rem; margin: 0;">{icons[cluster]}</h1>
            <h3 style="margin: 5px 0; color: {colors[cluster]};">Segment {cluster}</h3>
            <p style="margin: 0; font-weight: bold;">{names[cluster]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Check if 'Size' column exists
        size_text = ""
        if 'Size' in profiles.columns:
            size_text = f" | Size: {int(row['Size'])} customers"
        
        st.markdown(f"""
        <div style="background: #1E2128; border-radius: 12px; padding: 15px; height: 100%;">
            <b style="color: {colors[cluster]};">📋 Profile:</b><br>
            Income: {row['Annual_Income_k']:.1f}k | 
            Spending: {row['Spending_Score']:.1f}{size_text}
            <br><br>
            <b style="color: {colors[cluster]};">🚀 Strategy:</b><br>
            {get_strategy(cluster)}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")