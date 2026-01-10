
import logging
import os
import sys
import pandas as pd
import joblib
import numpy as np
from src.data.preprocessing import load_and_preprocess
from src.data.batch_generator import generate_batches
from src.model.train import train_model
from src.model.drift_detector import detect_drift
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, PSI_THRESHOLD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    try:
        # 1. Preprocess & Batches
        logger.info("Step 1: Preprocessing...")
        df, scaler, le = load_and_preprocess()
        batches = generate_batches(df, n_batches=5)
        for i, batch in enumerate(batches):
            batch.to_csv(f"data/batches/batch_{i}.csv", index=False)
        logger.info(f"Batches generated: {len(batches)} files, churn ~{df[TARGET].mean():.2%}")

        # 2. Baseline Train
        logger.info("Step 2: Training baseline...")
        baseline_results = train_model('data/batches/batch_0.csv', 'baseline')
        f1_base = baseline_results['f1']
        auc_base = baseline_results['auc']
        model_base = baseline_results['model']
        logger.info(f"Baseline: F1 {f1_base:.3f}, AUC {auc_base:.3f}")

        # 3. Drift Detection (sim on batch_2)
        logger.info("Step 3: Detecting drift...")
        baseline_df = pd.read_csv("data/batches/batch_0.csv")
        current_df = pd.read_csv("data/batches/batch_2.csv")
        drifted_df = current_df.copy()
        drifted_df['CustomerCareCalls'] += np.abs(drifted_df['CustomerCareCalls']) * 0.15  # Sim drift
        drifts, has_drift = detect_drift(baseline_df, drifted_df)
        logger.info(f"Drift detected: {has_drift} (PSI > {PSI_THRESHOLD})")

        # 4. Retrain if Drift
        if has_drift:
            logger.info("Step 4: Retraining...")
            recent_data = pd.concat([pd.read_csv(f"data/batches/batch_{i}.csv") for i in range(1, 5)], ignore_index=True)
            recent_path = 'data/batches/recent_concat.csv'
            recent_data.to_csv(recent_path, index=False)
            retrain_results = train_model(recent_path, 'retrained')
            f1_re = retrain_results['f1']
            auc_re = retrain_results['auc']
            model_re = retrain_results['model']
            logger.info(f"Retrained: F1 {f1_re:.3f}, AUC {auc_re:.3f}")
            
            # Deltas
            delta_f1 = f1_re - f1_base
            delta_auc = auc_re - auc_base
            logger.info(f"Deltas: F1 +{delta_f1:+.3f}, AUC +{delta_auc:+.3f}")
            print(f"Pipeline Complete: Baseline F1 {f1_base:.3f} → Retrained {f1_re:.3f} (delta +{delta_f1:+.3f})")
        else:
            logger.info("No drift — baseline stable.")
            print("Pipeline Complete: No retrain needed.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()