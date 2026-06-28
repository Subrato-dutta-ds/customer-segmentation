import os
import sys
import streamlit as st

def ensure_model_exists():
    """Check if model files exist. If not, run training on the cloud."""
    model_path = 'models/kmeans_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.warning("⏳ Model files not found. Training the model on the fly (this happens only once on first load)...")
        
        # Import and run training
        sys.path.append('.')
        from src.train import main as train_main
        
        # Run training
        train_main()
        
        st.success("✅ Model trained successfully! Refreshing the app...")
        st.rerun()