import json

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj

def build_llm_summary(payload):
    return {
        "dataset": {
            "rows": payload["schema"]["rows"],
            "cols": payload["schema"]["cols"]
        },
        "performance": {
            "accuracy": payload["metrics"]["accuracy"],
            "fraud_precision": payload["metrics"]["1"]["precision"],
            "fraud_recall": payload["metrics"]["1"]["recall"]
        },
        "business": payload.get("cost_analysis", {}),
        "top_features": list(payload["feature_importance"].keys())[:5],
        "drift_top": dict(list(payload["drift"].items())[:5])
    }
