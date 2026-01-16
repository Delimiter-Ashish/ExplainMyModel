import numpy as np
from sklearn.calibration import calibration_curve

def class_imbalance(y):
    return y.value_counts(normalize=True).to_dict()

def drift_detection(X_train, X_test):
    drift = {}
    for col in X_train.columns:
        drift[col] = float(abs(X_train[col].mean() - X_test[col].mean()))
    return drift

def cost_analysis(y_true, y_pred, cost_fn=500, cost_fp=5):
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return {
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "estimated_cost": float(fn * cost_fn + fp * cost_fp)
    }

def calibration_analysis(y_true, probs):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
    return {
        "mean_predicted": mean_pred.tolist(),
        "fraction_positive": frac_pos.tolist()
    }
