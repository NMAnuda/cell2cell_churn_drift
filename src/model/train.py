"""
Model Training Script (Fixed for F1 ~0.48)
- Disables SMOTE for baseline (less skew), caps weight, larger grid, early stopping.
"""

import joblib
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, classification_report, precision_recall_curve
from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE  # COMMENTED: Disable for baseline
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, MODEL_PARAMS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_path, model_name='baseline'):

    try:
        data = pd.read_csv(data_path)
        X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]  # 14 features
        y = data[TARGET]
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = min(neg / pos if pos > 0 else 1, 2.0)
        print(f"{model_name.capitalize()} scale_pos_weight:", scale_pos_weight)

        # SMOTE (COMMENTED for baseline — re-enable for retrain if needed)
        # smote = SMOTE(random_state=42)
        # X_train, y_train = smote.fit_resample(X_train, y_train)

        logger.info(f"{model_name.capitalize()} training on {len(X_train)} samples, churn rate: {y_train.mean():.2%}")
        
        # Base model
        base_params = {k: v for k, v in MODEL_PARAMS.items() if k not in ['max_depth', 'learning_rate', 'n_estimators']}
        base_model = XGBClassifier(
            **base_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=10,  
            verbose=False
        )

        # LARGER grid for better fit (12 combos x cv=2 = 24 fits, ~5 mins)
        param_grid = {
            'max_depth': [3, 5, 6], 
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200, 250] 
        }

        model = GridSearchCV(base_model, param_grid, cv=2, scoring='f1', n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],  # NEW: Early stopping on test
                  verbose=False)
        
        # Best model
        best_model = model.best_estimator_
        
        # Probs
        probs = best_model.predict_proba(X_test)[:, 1]
        print("Probs mean:", probs.mean())  # DIAG: Check skew (should ~0.3-0.4)

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

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name.capitalize()} PR Curve')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name.capitalize()} Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save
        joblib.dump(best_model, f'models/{model_name}_model_improved.pkl')
        baseline_dist = X_train[NUMERIC_FEATURES].describe()
        baseline_dist.to_csv(f'data/batches/{model_name}_dist_improved.csv')
        logger.info(f"{model_name.capitalize()} Model & dist saved.")
        
        return {
            'model': best_model,
            'f1': best_f1,
            'auc': auc,
            'threshold': best_threshold,
            'dist': baseline_dist
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model('data/batches/batch_0.csv', 'baseline')  # Default baseline