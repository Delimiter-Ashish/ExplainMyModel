import shap
import numpy as np

def shap_explain(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = dict(
        zip(X.columns, np.abs(shap_values).mean(axis=0))
    )

    return dict(sorted(importance.items(), key=lambda x: -x[1]))
