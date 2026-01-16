from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def build_model(model_type: str, config: dict):
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 5),
            eval_metric="logloss"
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.get("n_estimators", 300)
        )

    raise ValueError(f"Unsupported model type: {model_type}")
