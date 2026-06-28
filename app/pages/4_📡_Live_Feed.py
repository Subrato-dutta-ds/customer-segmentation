import streamlit as st
import pandas as pd
import time
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.simulator import CustomerSimulator
from src.predict import load_models

st.set_page_config(page_title="Live Customer Feed", layout="wide")
st.title("📡 Live Customer Data Stream")
st.markdown("**Simulated real-time customer arrivals** — watch new customers flow in and get automatically segmented!")

# --- Initialize Session State ---
if 'live_df' not in st.session_state:
    st.session_state.live_df = None
if 'live_sim' not in st.session_state:
    st.session_state.live_sim = CustomerSimulator()
    initial_df = st.session_state.live_sim.stream_to_dataframe(100)
    st.session_state.live_df = initial_df
if 'live_count' not in st.session_state:
    st.session_state.live_count = 0

@st.cache_resource
def load_ml_models():
    try:
        model, scaler = load_models()
        return model, scaler
    except:
        return None, None

model, scaler = load_ml_models()

# Segment Colors
segment_colors = ['#6C63FF', '#FF6584', '#00D4FF', '#00E676', '#FFB74D']

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Live Feed Controls")
batch_size = st.sidebar.slider("Customers per batch", min_value=1, max_value=20, value=5)
auto_stream = st.sidebar.checkbox("Auto-stream every 2 seconds", value=False)

# --- Main Buttons Row ---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("➕ Add New Customer", use_container_width=True):
        new_customer = st.session_state.live_sim.generate_customer()
        new_df = pd.DataFrame([new_customer])
        st.session_state.live_df = pd.concat([st.session_state.live_df, new_df], ignore_index=True)
        st.session_state.live_count += 1
        st.rerun()

with col2:
    if st.button(f"📦 Add Batch ({batch_size})", use_container_width=True):
        batch = st.session_state.live_sim.generate_batch(batch_size)
        new_df = pd.DataFrame(batch)
        st.session_state.live_df = pd.concat([st.session_state.live_df, new_df], ignore_index=True)
        st.session_state.live_count += len(batch)
        st.rerun()

with col3:
    if st.button("🔄 Reset & Regenerate", use_container_width=True):
        st.session_state.live_sim = CustomerSimulator()
        st.session_state.live_df = st.session_state.live_sim.stream_to_dataframe(100)
        st.session_state.live_count = 0
        st.rerun()

# --- Auto-stream Logic ---
if auto_stream:
    placeholder = st.empty()
    for i in range(5):
        new_customer = st.session_state.live_sim.generate_customer()
        new_df = pd.DataFrame([new_customer])
        st.session_state.live_df = pd.concat([st.session_state.live_df, new_df], ignore_index=True)
        st.session_state.live_count += 1
        placeholder.write(f"⏳ Streaming... Added {st.session_state.live_count} customers so far")
        time.sleep(2)
    placeholder.empty()
    st.rerun()

# --- Display the Data ---
if st.session_state.live_df is not None and not st.session_state.live_df.empty:
    df = st.session_state.live_df.copy()
    
    # Assign clusters using 3 features
    if model and scaler and all(col in df.columns for col in ['Annual_Income_k', 'Spending_Score', 'Age']):
        X = df[['Annual_Income_k', 'Spending_Score', 'Age']]
        X_scaled = scaler.transform(X)
        df['Cluster'] = model.predict(X_scaled)
    
    # --- Metrics Row ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", len(df))
    col2.metric("🆕 New This Session", st.session_state.live_count)
    
    if 'Cluster' in df.columns:
        col3.metric("📊 Active Segments", df['Cluster'].nunique())
        most_common = df['Cluster'].mode().iloc[0] if not df['Cluster'].mode().empty else "-"
        col4.metric("🏆 Most Common", f"Cluster {most_common}")
    
    # --- Latest Customers Table ---
    st.subheader("🆕 Latest Arrivals")
    latest = df.tail(10).sort_values('CustomerID', ascending=False)
    display_cols = ['CustomerID', 'Age', 'Annual_Income_k', 'Spending_Score']
    if 'Cluster' in df.columns:
        display_cols.append('Cluster')
    st.dataframe(latest[display_cols], use_container_width=True)
    
    # --- Live Scatter Plot with Gradient Colors ---
    st.subheader("📊 Live Customer Distribution")
    if 'Cluster' in df.columns:
        fig = px.scatter(
            df,
            x='Annual_Income_k',
            y='Spending_Score',
            color='Cluster',
            color_discrete_sequence=segment_colors,
            hover_data=['CustomerID', 'Age'],
            title='Customer Segments (Live Data)'
        )
    else:
        fig = px.scatter(
            df,
            x='Annual_Income_k',
            y='Spending_Score',
            hover_data=['CustomerID', 'Age'],
            title='Customer Segments (Live Data)'
        )
    fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Download Button ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Live Dataset as CSV",
        data=csv,
        file_name=f"live_customers_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv'
    )
    
else:
    st.info("💡 Click 'Add New Customer' or 'Add Batch' to start building your live dataset!")

# --- Session Stats in Sidebar ---
st.sidebar.subheader("📊 Session Stats")
if st.session_state.live_df is not None:
    st.sidebar.write(f"**Total:** {len(st.session_state.live_df)} customers")
    st.sidebar.write(f"**New:** {st.session_state.live_count}")
    if 'Cluster' in st.session_state.live_df.columns:
        st.sidebar.write("---")
        st.sidebar.write("**Segment Distribution:**")
        clusters = st.session_state.live_df['Cluster'].value_counts().sort_index()
        for cluster, count in clusters.items():
            pct = (count / len(st.session_state.live_df)) * 100
            color = segment_colors[cluster] if cluster < len(segment_colors) else "#FFFFFF"
            st.sidebar.markdown(f'<span style="color:{color};">●</span> Cluster {cluster}: {count} ({pct:.1f}%)', unsafe_allow_html=True)