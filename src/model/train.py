"""
Model Training Script (Flexible for Baseline/Retrain with Versioning)
- Trains XGBoost with SMOTE, dynamic weight, reduced tuning, threshold opt for F1.
- Logs metadata (version, F1, params) to JSON.
- Usage: python src/model/train.py (baseline default).
"""

import joblib
import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, classification_report, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, MODEL_PARAMS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_path, model_name='baseline'):
    try:
        data = pd.read_csv(data_path)
        X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = data[TARGET]
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1
        print(f"{model_name.capitalize()} scale_pos_weight:", scale_pos_weight)

        # SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        logger.info(f"{model_name.capitalize()} training on {len(X_train)} samples, churn rate: {y_train.mean():.2%}")
        
        # Base model
        base_params = {k: v for k, v in MODEL_PARAMS.items() if k not in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree', 'min_child_weight', 'gamma']}
        base_model = XGBClassifier(
            **base_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )

        # Reduced param_grid for speed
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200]
        }

        model = GridSearchCV(base_model, param_grid, cv=2, scoring='f1', n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Best model
        best_model = model.best_estimator_
        
        # Probs
        probs = best_model.predict_proba(X_test)[:, 1]

        # Threshold opt
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = [f1_score(y_test, (probs >= t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)

        print(f"{model_name.capitalize()} Best threshold:", best_threshold)
        print(f"{model_name.capitalize()} Best F1:", best_f1)

        # Final preds
        y_pred = (probs >= best_threshold).astype(int)

        # AUC
        auc = roc_auc_score(y_test, probs)

        # Report
        print(f"\n{model_name.capitalize()} Classification Report (Optimized Threshold):\n")
        print(classification_report(y_test, y_pred))
        print(f"{model_name.capitalize()} AUC:", auc)

        logger.info(f"{model_name.capitalize()} Best params: {model.best_params_}")
        print(f"{model_name.capitalize()} F1: {best_f1:.3f}, AUC: {auc:.3f}")

        # PR Curve - Save but don't show
        precision, recall, _ = precision_recall_curve(y_test, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name.capitalize()} PR Curve', linewidth=2)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(f"{model_name.capitalize()} Precision-Recall Curve", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot instead of showing it
        pr_curve_path = f'models/{model_name}_pr_curve.png'
        plt.savefig(pr_curve_path, dpi=150)
        plt.close()
        logger.info(f"PR curve saved to {pr_curve_path}")

        # Save model and distribution
        joblib.dump(best_model, f'models/{model_name}_model_improved.pkl')
        baseline_dist = X_train[NUMERIC_FEATURES].describe()
        baseline_dist.to_csv(f'data/batches/{model_name}_dist_improved.csv')
        logger.info(f"{model_name.capitalize()} Model & dist saved.")
        
        # Versioning
        metadata = {
            'version': f"{model_name}_v{datetime.now().strftime('%Y%m%d_%H%M')}",
            'f1': best_f1,
            'auc': auc,
            'threshold': best_threshold,
            'params': model.best_params_,
            'churn_rate': y.mean(),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(f'models/{model_name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Versioned: {metadata['version']} (F1 {best_f1:.3f})")
        
        return {
            'model': best_model,
            'f1': best_f1,
            'auc': auc,
            'threshold': best_threshold,
            'dist': baseline_dist,
            'metadata': metadata,
            'pr_curve_path': pr_curve_path
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model('data/batches/batch_0.csv', 'baseline')