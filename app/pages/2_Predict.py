import streamlit as st
import pandas as pd
import joblib
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Predict Segment", layout="wide")
st.title("🔮 Predict Customer Segment")

@st.cache_resource
def load_model():
    model = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Segment Colors
segment_colors = ['#6C63FF', '#FF6584', '#00D4FF', '#00E676', '#FFB74D']
segment_names = ['💎 Premium VIP', '📈 Standard', '🛒 Budget', '🔥 Impulse Spender', '👴 Retired']

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Customer Details")
    income = st.slider("💰 Annual Income (k$)", 10, 150, 60)
    spending = st.slider("🛍️ Spending Score (1-100)", 1, 100, 50)
    age = st.slider("👤 Age", 18, 75, 30)

    if st.button("🚀 Predict Segment", type="primary", use_container_width=True):
        X = np.array([[income, spending, age]])
        X_scaled = scaler.transform(X)
        cluster = model.predict(X_scaled)[0]
        
        color = segment_colors[cluster]
        name = segment_names[cluster]

        st.markdown("---")
        st.subheader("✅ Prediction Result")
        
        st.markdown(f"""
        <div style="background: rgba(26, 26, 46, 0.9); border-left: 8px solid {color}; padding: 2rem; border-radius: 16px; margin-top: 10px;">
            <h2 style="margin:0; color: {color};">{name}</h2>
            <p style="margin:5px 0; color: #BBBBBB;">
                This customer belongs to <b style="color: {color};">Cluster {cluster}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Strategy Card
        strategies = {
            0: "🌟 **VIP Treatment**: Exclusive perks, loyalty cashback, and early access to new products.",
            1: "📈 **Upsell Opportunity**: Personalized recommendations and product bundles.",
            2: "🛒 **Value Focus**: Budget-friendly packages, bulk discounts, and loyalty points.",
            3: "🔥 **Engagement**: Social media campaigns, flash sales, and referral bonuses.",
            4: "👴 **Trust Building**: Easy-to-use interfaces, senior discounts, and personalized support."
        }
        
        st.info(strategies.get(cluster, "Standard retention marketing."))

with col2:
    st.subheader("📊 Segment Reference")
    st.markdown("""
    | Cluster | Persona | Strategy |
    |---------|---------|----------|
    | 0 🟣 | 💎 Premium VIP | VIP Perks |
    | 1 🔴 | 📈 Standard | Upsell |
    | 2 🔵 | 🛒 Budget | Value Deals |
    | 3 🟢 | 🔥 Impulse | Flash Sales |
    | 4 🟠 | 👴 Retired | Trust Building |
    """)