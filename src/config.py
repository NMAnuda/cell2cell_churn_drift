# Configuration for Cell2Cell Churn Drift Pipeline
import os


DATA_PATH = "data/"
RAW_DATA = os.path.join(DATA_PATH, "raw/cell2cellholdout.csv") 
PROCESSED_DATA = os.path.join(DATA_PATH, "processed/churn_processed.csv")

# Drift Thresholds
PSI_THRESHOLD = 0.25
KS_THRESHOLD = 0.1

# Model Hyperparams
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'scale_pos_weight': 5  # For ~16% churn imbalance
}

# Features (expanded for full dataset — all exist)
NUMERIC_FEATURES = [
    'MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
    'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls',
    'PercChangeMinutes', 'PercChangeRevenues', 'CustServCalls', 'HandsetPrice'
]  # 'CustServCalls', 'HandsetPrice' are common in full set
CATEGORICAL_FEATURES = ['IncomeGroup', 'OwnsMotorcycle']  # Low-cardinality from EDA
TARGET = 'Churn'  # Now 0/1 in full data

# AWS (later)
S3_BUCKET = 'your-churn-drift-bucket'
S3_RAW_PREFIX = 'raw/'
S3_LOGS_PREFIX = 'inference-logs/'
REGION = 'us-east-1'