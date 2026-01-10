import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.data.batch_generator import generate_batches  # FIXED: Absolute import
from src.config import RAW_DATA, PROCESSED_DATA, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET

def load_and_preprocess():
    df = pd.read_csv(RAW_DATA)
    print(f"Raw shape: {df.shape}")
    print("Raw Churn sample:", df[TARGET].head().tolist())
    
    # Churn mapping
    df[TARGET] = df[TARGET].map({'Yes': 1, 'No': 0, 1: 1, 0: 0, 'True': 1, 'False': 0, np.nan: 0})
    df[TARGET] = df[TARGET].fillna(0).astype(int)
    print(f"Processed Churn distribution:\n{df[TARGET].value_counts(normalize=True)}")
    
    # FIXED Numerics: Convert to numeric first (coerce strings to NaN), then fill median
    existing_numerics = [col for col in NUMERIC_FEATURES if col in df.columns]
    print(f"Using {len(existing_numerics)} numerics")
    for col in existing_numerics:
        # Force numeric (strings like '30' → 30.0, 'Unknown' → NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
        print(f"{col} dtype after fix: {df[col].dtype}")  # Debug: Should be float64
    
    # Categoricals
    existing_cats = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    le = LabelEncoder()
    for col in existing_cats:
        df[col] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
    
    # Scale
    scaler = StandardScaler()
    if existing_numerics:
        df[existing_numerics] = scaler.fit_transform(df[existing_numerics])
    
    # Drop ID
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
    
    # Save
    os.makedirs(os.path.dirname(PROCESSED_DATA), exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"Processed saved: {PROCESSED_DATA}")
    print("Processed head (Churn + first few):")
    print(df[[TARGET] + NUMERIC_FEATURES[:3]].head())
    
    return df, scaler, le

if __name__ == "__main__":
    df, scaler, le = load_and_preprocess()
    batches = generate_batches(df, n_batches=5)
    for i, batch in enumerate(batches):
        batch.to_csv(f"data/batches/batch_{i}.csv", index=False)
    print("Preprocessing + batches complete!")