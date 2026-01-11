"""
Local MLOps Simulation: Full Churn Pipeline
- Preprocess → Batches → Train → Drift Detect → Retrain → Log MLflow → Alert → Schedule
- Usage: python run_pipeline.py
"""

from sklearn.metrics import precision_recall_curve
import logging
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import schedule
import time
from datetime import datetime

from src.data.preprocessing import load_and_preprocess
from src.data.batch_generator import generate_batches
from src.model.train import train_model
from src.model.drift_detector import detect_drift
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, PSI_THRESHOLD

# =========================
# MLflow Setup
# =========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Churn Drift Pipeline")

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =========================
# Pipeline Function
# =========================
def run_pipeline():
    logger.info("Pipeline started...")

    try:
        with mlflow.start_run(run_name=f"churn_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            # 1. Preprocess
            logger.info("Step 1: Preprocessing...")
            df, scaler, le = load_and_preprocess()

            batches = generate_batches(df, n_batches=5)
            for i, batch in enumerate(batches):
                batch.to_csv(f"data/batches/batch_{i}.csv", index=False)

            logger.info(f"Batches generated: {len(batches)}")

            # Log parameters EARLY
            mlflow.log_param("n_batches", 5)
            mlflow.log_param("psi_threshold", PSI_THRESHOLD)
            mlflow.log_param("target", TARGET)
            logger.info("Parameters logged")

            # 2. Baseline Train
            logger.info("Step 2: Training baseline...")
            baseline_results = train_model("data/batches/batch_0.csv", "baseline")

            f1_base = baseline_results["f1"]
            auc_base = baseline_results["auc"]
            best_model = baseline_results["model"]
            threshold_base = baseline_results["threshold"]

            logger.info(f"Baseline: F1={f1_base:.3f}, AUC={auc_base:.3f}")

            # Log baseline metrics immediately
            mlflow.log_metric("baseline_f1", f1_base)
            mlflow.log_metric("baseline_auc", auc_base)
            mlflow.log_param("baseline_threshold", threshold_base)
            logger.info("Baseline metrics logged")

            # 3. Drift Detection
            logger.info("Step 3: Detecting drift...")
            baseline_df = pd.read_csv("data/batches/batch_0.csv")
            current_df = pd.read_csv("data/batches/batch_2.csv")

            drifted_df = current_df.copy()
            drifted_df["CustomerCareCalls"] += np.abs(drifted_df["CustomerCareCalls"]) * 0.15

            drifts, has_drift = detect_drift(baseline_df, drifted_df)
            drift_psi_avg = np.mean([d["psi"] for d in drifts.values()])

            logger.info(f"Drift detected: {has_drift}, PSI avg: {drift_psi_avg:.3f}")

            # Log drift metrics
            mlflow.log_metric("drift_detected", int(has_drift))
            mlflow.log_metric("drift_psi_avg", drift_psi_avg)
            mlflow.log_param("drift_threshold", PSI_THRESHOLD)
            logger.info("Drift metrics logged")

            # 4. Retrain if Drift
            f1_re = f1_base
            auc_re = auc_base
            model_re = best_model
            delta_f1 = 0
            threshold_re = threshold_base

            if has_drift:
                logger.info("Step 4: Retraining...")

                recent_data = pd.concat(
                    [pd.read_csv(f"data/batches/batch_{i}.csv") for i in range(1, 5)],
                    ignore_index=True
                )

                recent_path = "data/batches/recent_concat.csv"
                recent_data.to_csv(recent_path, index=False)

                retrain_results = train_model(recent_path, "retrained")

                f1_re = retrain_results["f1"]
                auc_re = retrain_results["auc"]
                model_re = retrain_results["model"]
                threshold_re = retrain_results["threshold"]

                delta_f1 = f1_re - f1_base

                logger.info(f"Retrained: F1={f1_re:.3f}, AUC={auc_re:.3f}")

                # Log retrain metrics
                mlflow.log_metric("retrained_f1", f1_re)
                mlflow.log_metric("retrained_auc", auc_re)
                mlflow.log_metric("delta_f1", delta_f1)
                mlflow.log_param("retrained_threshold", threshold_re)
                mlflow.log_param("retrained", True)
                logger.info("Retrain metrics logged")
            else:
                mlflow.log_param("retrained", False)

            # 5. Log Models
            logger.info("Logging models to MLflow...")
            mlflow.xgboost.log_model(best_model, "baseline_model")
            mlflow.xgboost.log_model(model_re, "retrained_model")
            logger.info("Models logged")

            # 6. PR Curve
            logger.info("Generating PR curve...")
            X_test = baseline_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
            y_test = baseline_df[TARGET]

            probs_base = best_model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, probs_base)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label="Baseline PR Curve", linewidth=2)
            plt.xlabel("Recall", fontsize=12)
            plt.ylabel("Precision", fontsize=12)
            plt.title("Precision-Recall Curve", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("pr_curve.png", dpi=150)
            plt.close()

            mlflow.log_artifact("pr_curve.png")
            logger.info("PR curve logged")

            # 7. Summary metrics
            mlflow.log_metric("final_f1", f1_re)
            mlflow.log_metric("final_auc", auc_re)
            mlflow.log_metric("improvement", delta_f1)

            logger.info("=" * 60)
            logger.info("PIPELINE SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Baseline F1: {f1_base:.3f}, AUC: {auc_base:.3f}")
            logger.info(f"Retrained F1: {f1_re:.3f}, AUC: {auc_re:.3f}")
            logger.info(f"Improvement: {delta_f1:.3f}")
            logger.info(f"Drift Detected: {has_drift} (PSI: {drift_psi_avg:.3f})")
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            logger.info("=" * 60)

        logger.info(" Pipeline finished successfully. MLflow run closed.")
        print("\n All metrics logged to MLflow!")
        print(f"🔍 View results: mlflow ui --port 5000")
        print(f"   Then navigate to: http://localhost:5000")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# =========================
# Scheduler
# =========================
def job():
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Scheduled job failed: {e}")


if __name__ == "__main__":

    print("Running pipeline immediately...")
    
    job()


    print("Scheduler started (daily at 02:00)")
    print("  Press Ctrl+C to stop")

    schedule.every().day.at("02:00").do(job)

    while True:
        schedule.run_pending()
        time.sleep(60)