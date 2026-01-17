# Automated Concept Drift Detection & Model Retraining System (MLOps)

## Overview
This project implements an end-to-end **MLOps pipeline** for customer churn prediction with automated **concept drift detection**, **model retraining**, **versioning**, and **monitoring**. It simulates a real-world production environment using time-based data batches and supports both **local execution** and **AWS-integrated deployment**.

The system is designed to continuously monitor incoming data, detect distribution shifts, trigger retraining when necessary, and deploy improved models automatically.

---

##  Key Features

- End-to-end churn prediction pipeline
- Time-based batch simulation
- Automated preprocessing & feature engineering
- Baseline model training (XGBoost)
- Concept drift detection using PSI
- Automatic retraining on recent data
- Model versioning & metadata tracking
- MLflow experiment tracking & artifact logging
- Flask API for real-time predictions
- Scheduler for periodic execution
- Dual mode:
  - Local-safe mode
  - AWS-integrated mode (EC2, S3, Lambda)

---

##  Use Case

**Customer Churn Prediction for a Telecom Company**

The system monitors customer behavior over time and automatically adapts when data distributions change, ensuring the model remains accurate and reliable.

---

##  Architecture

    User / Scheduler
    |
    v
    EC2 Instance
    |
    |----> S3 (Data, Models, Artifacts)
    |
    |----> Lambda (Drift Alerts / Triggers)
    |
    v
    MLflow UI (Metrics & Versions)

---

##  Project Structure

    cell2cell_churn_drift/
    │
    ├── data/
    │   ├── raw/                          # Original dataset files (CSV)
    │   ├── processed/                    # Cleaned & feature-engineered datasets
    │   └── batches/                      # Time-split batches for drift simulation
    │
    ├── notebooks/
    │   ├── 01_data_exploration.ipynb     # EDA, feature distributions
    │   ├── 02_baseline_model.ipynb       # Train initial churn model
    │   ├── 03_drift_analysis.ipynb       # PSI, KS test analysis
    │   └── 04_model_retraining.ipynb     # Retraining experiments
    │
    ├── src/
    │   ├── data/
    │   │   ├── preprocessing.py          # Cleaning, feature encoding, scaling
    │   │   ├── batch_generator.py        # Split data into simulated time batches
    │   │   └── utils.py                  # Helper functions
    │   │
    │   ├── model/
    │   │   ├── train.py                  # Model training script
    │   │   ├── evaluate.py               # Model evaluation metrics
    │   │   ├── inference.py              # Make predictions on new data
    │   │   └── drift_detector.py         # PSI, KS, label drift calculations
    │   │
    │   ├── aws/
    │   │   ├── s3_upload.py              # Upload raw/processed data to S3
    │   │   ├── lambda_handler.py         # Lambda logic for drift detection
    │   │   ├── trigger_retrain.py        # Trigger SageMaker retraining
    │   │   └── deploy_model.py           # Deploy trained model to endpoint
    │   │
    │   └── config.py                     # Global paths, thresholds, and constants
    │
    ├── requirements.txt                  # Python dependencies
    ├── README.md                         # Project overview and documentation
    ├── .env                              # AWS credentials (gitignore'd)
    ├── .gitignore
    ├── serve-model.py
    └── run_pipeline.py                   # Master script (toggle AWS)


---

## Pipeline Flow

1. Data ingestion & preprocessing
2. Batch generation (time-based simulation)
3. Baseline model training
4. Drift detection using PSI
5. Retraining on recent batches (if drift detected)
6. Model comparison & improvement tracking
7. Model versioning & artifact logging
8. Deployment-ready model saving
9. API serving (Flask)
10. Scheduled automation

---

##  Drift Detection

The system detects:
- Feature distribution shifts (PSI)
- Label distribution changes
- Performance degradation

If drift exceeds the threshold:
- Retraining is triggered
- New model is evaluated
- Best model is versioned & logged

---

##  AWS Integration (Optional)

This project supports AWS-based execution using:

| Service | Purpose |
|--------|--------|
| EC2 | Runs the full pipeline |
| S3 | Stores datasets, models, artifacts |
| Lambda | Drift alerts & triggers |
| MLflow | Experiment tracking |

### Toggle Mode

```python
USE_AWS = False  # Local mode
USE_AWS = True   # AWS-integrated mode
```

## How to Run Locally

pip install -r requirements.txt
python run_pipeline.py
# MLflow UI:
mlflow ui --port 5000

## Technologies Used

Python | XGBoost | Scikit-learn | Pandas, NumPy | MLflow | Flask | Schedule | Matplotlib | AWS (EC2, S3, Lambda) | Boto3

![](./pictrure/image.png)
