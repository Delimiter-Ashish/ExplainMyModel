import pandas as pd

def group_fairness(y_true, y_pred, sensitive_series):
    results = {}

    for group in sensitive_series.unique():
        mask = sensitive_series == group
        acc = (y_true[mask] == y_pred[mask]).mean()
        results[str(group)] = float(acc)

    gap = max(results.values()) - min(results.values())

    return {
        "group_accuracy": results,
        "bias_gap": float(gap),
        "risk_level": "High" if gap > 0.1 else "Medium" if gap > 0.05 else "Low"
    }
