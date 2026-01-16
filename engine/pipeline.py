import os
import json
from sklearn.model_selection import train_test_split

from engine.data_loader import load_csv
from engine.schema import analyze_schema
from engine.preprocessing import preprocess
from engine.modeling import build_model
from engine.evaluation import evaluate_model
from engine.diagnostics import (
    class_imbalance,
    drift_detection,
    cost_analysis,
    calibration_analysis
)
from engine.explainability import shap_explain
from engine.reporting import to_json_safe, build_llm_summary

def run_pipeline(
    csv_source: str,
    target: str,
    model_type: str,
    model_config: dict,
    output_dir: str = "artifacts"
):
    os.makedirs(output_dir, exist_ok=True)

    df = load_csv(csv_source)
    schema = analyze_schema(df)

    df, encoders = preprocess(df)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model(model_type, model_config)
    model.fit(X_train, y_train)

    eval_res = evaluate_model(model, X_test, y_test)
    preds = eval_res["predictions"]
    metrics = eval_res["metrics"]

    payload = {
        "schema": schema,
        "metrics": metrics,
        "class_imbalance": class_imbalance(y),
        "drift": drift_detection(X_train, X_test),
        "cost_analysis": cost_analysis(y_test, preds),
        "feature_importance": shap_explain(model, X_test)
    }

    safe_payload = to_json_safe(payload)

    with open(f"{output_dir}/payload.json", "w") as f:
        json.dump(safe_payload, f, indent=2)

    return safe_payload
