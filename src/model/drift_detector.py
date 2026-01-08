import numpy as np
from scipy.stats import ks_2samp
import warnings
from src.config import PSI_THRESHOLD, KS_THRESHOLD, NUMERIC_FEATURES

def calculate_psi(baseline, current, bins=10, epsilon=1e-15):
    """Population Stability Index (PSI) for drift detection — robust to zeros"""
    def get_buckets(data, bins):
        hist, _ = np.histogram(data, bins=bins, density=True)
        # Add epsilon to avoid log(0); clip to prevent overflow
        return np.clip(hist + epsilon, epsilon, 1.0)
    
    baseline_buckets = get_buckets(baseline, bins)
    current_buckets = get_buckets(current, bins)
    
    # PSI calc with  log 
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        ratio = current_buckets / baseline_buckets
        psi = np.sum((current_buckets - baseline_buckets) * np.log(ratio))
    
    # Clip extreme values (inf/NaN → treat as high drift, but cap for sanity)
    if np.isinf(psi) or np.isnan(psi):
        psi = 1.0  # Arbitrary high value to flag drift
    psi = np.clip(psi, 0, 10)  # Cap at 10 (very drifted)
    
    return psi

def detect_drift(baseline_df, current_df):
    """Run PSI on numerics + KS on all — robust version"""
    drifts = {}
    for feature in NUMERIC_FEATURES:
        if feature not in baseline_df.columns or feature not in current_df.columns:
            continue  # Skip missing
        
        psi = calculate_psi(baseline_df[feature], current_df[feature])
        ks_stat, p_value = ks_2samp(baseline_df[feature], current_df[feature])
        
        drifts[feature] = {
            'psi': psi,
            'ks_pvalue': p_value,
            'drift_detected': (psi > PSI_THRESHOLD) or (p_value < KS_THRESHOLD)
        }
        print(f"{feature}: PSI={psi:.3f}, KS p={p_value:.3f}, Drift={drifts[feature]['drift_detected']}")  # Debug (remove later)
    
    overall_drift = any(d['drift_detected'] for d in drifts.values())
    return drifts, overall_drift