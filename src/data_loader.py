import pandas as pd
import os

def load_data(filepath='data/raw/Mall_Customers.csv', source='simulated', num_customers=200):
    """
    Load customer data from CSV or generate simulated data.
    
    Parameters:
    - filepath: Path to CSV file (if source='csv')
    - source: 'csv' or 'simulated'
    - num_customers: Number of customers to generate (if source='simulated')
    """
    if source == 'simulated':
        print(f"🔄 Generating {num_customers} simulated customers...")
        # We import here to avoid circular imports
        from src.simulator import CustomerSimulator
        sim = CustomerSimulator()
        df = sim.stream_to_dataframe(num_customers)
        print(f"✅ Loaded {len(df)} simulated customers!")
        return df, sim
    
    elif source == 'csv':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}. Please download it first.")
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded from CSV! Shape: {df.shape}")
        return df, None
    
    else:
        raise ValueError(f"Unknown source: {source}. Use 'csv' or 'simulated'.")

def clean_data(df):
    """
    Basic cleaning: ensure no missing values and standardise column names.
    """
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("⚠️ Missing values found. Dropping rows...")
        df = df.dropna()
    
    # Rename columns to standardised names for easier access
    if 'Annual Income (k$)' in df.columns:
        df.rename(columns={
            'Annual Income (k$)': 'Annual_Income_k',
            'Spending Score (1-100)': 'Spending_Score'
        }, inplace=True)
    
    print(f"✅ Data cleaned. Shape: {df.shape}")
    return df

# Test block
if __name__ == "__main__":
    df, sim = load_data(source='simulated', num_customers=50)
    df = clean_data(df)
    print(df.head())