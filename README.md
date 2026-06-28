# 🛍️ Customer Segmentation Suite

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Silhouette](https://img.shields.io/badge/Silhouette_Score-0.53-brightgreen)]()

> **Real-time Customer Segmentation Engine using K-Means Clustering**

## 📌 Overview

This project is an **end-to-end machine learning solution** that segments customers into 5 distinct personas based on their **Income, Spending Score, and Age**. It features a **real-time data simulator** and an **interactive Streamlit dashboard** for live customer tracking and segment prediction.

### ✨ Key Features

- 🎯 **K-Means Clustering** with 5 actionable customer segments
- 📊 **Interactive Dashboard** with visualizations and metrics
- 🔮 **Real-time Prediction** for new customers
- 📡 **Live Data Feed** simulating customer arrivals
- 💡 **Personalized Business Strategies** for each segment
- 🐳 **Docker-ready** for easy deployment

### 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Silhouette Score** | **0.53** (Good) |
| **Optimal Clusters** | 5 |
| **Features Used** | Income, Spending, Age |

### 🧩 Customer Segments

| Cluster | Persona | Strategy |
|---------|---------|----------|
| 0 🟣 | **Premium VIP** | Exclusive perks, loyalty programs |
| 1 🔴 | **Standard** | Upselling, cross-selling |
| 2 🔵 | **Budget** | Value deals, bulk discounts |
| 3 🟢 | **Impulse Spender** | Flash sales, social media |
| 4 🟠 | **Retired** | Trust-building campaigns |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.10 |
| **ML Framework** | Scikit-learn (K-Means) |
| **Dashboard** | Streamlit |
| **Visualization** | Plotly, Matplotlib |
| **Data Processing** | Pandas, NumPy |
| **Containerization** | Docker |
| **Version Control** | Git |

---

## 📁 Project Structure
customer-segmentation/
├── .streamlit/
│ └── config.toml # Streamlit theme
├── app/
│ ├── app.py # Main app
│ └── pages/
│ ├── 1_Dashboard.py # Dashboard
│ ├── 2_Predict.py # Predict segment
│ ├── 3_Recommendations.py # Strategies
│ └── 4_Live_Feed.py # Live streaming
├── src/
│ ├── data_loader.py # Load data
│ ├── preprocess.py # Feature engineering
│ ├── train.py # Model training
│ ├── predict.py # Predictions
│ ├── simulator.py # Live data generator
│ └── recommendations.py # Business strategies
├── models/ # Saved models (gitignored)
├── data/ # Dataset (gitignored)
├── tests/ # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md

