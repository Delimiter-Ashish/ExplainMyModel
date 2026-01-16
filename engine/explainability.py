import shap
import numpy as np

def shap_explain(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle different SHAP output formats safely
    if isinstance(shap_values, list):
        # Binary classification → take positive class
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Ensure 2D: (samples, features)
    shap_vals = np.array(shap_vals)

    if shap_vals.ndim == 3:
        # (samples, features, classes) → average over classes
        shap_vals = shap_vals.mean(axis=2)

    # Compute scalar importance per feature
    feature_importance = np.abs(shap_vals).mean(axis=0)

    importance = {
        col: float(val)
        for col, val in zip(X.columns, feature_importance)
    }

    # Sort descending
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
