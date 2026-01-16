import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from engine.data_loader import load_csv
from engine.schema import analyze_schema
from engine.preprocessing import build_preprocessor
from engine.modeling import train_models
from engine.explainability import compute_shap
from engine.diagnostics import calibration_analysis
from engine.artifact_manager import save_json
from engine.experiment_tracker import log_experiment
from engine.llm_reasoner import generate_reasoned_audit

from monitoring.drift_monitor import ks_drift
from monitoring.alerting import generate_alerts


def run_pipeline(
    csv_source: str,
    target: str,
    sensitive_col: str = None,
    output_dir: str = "artifacts"
):
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------
    # 1) LOAD DATA
    # ---------------------------
    df = load_csv(csv_source)

    # Save schema (still useful)
    schema = analyze_schema(df)

    # ---------------------------
    # 2) PREPROCESS + SPLIT
    # ---------------------------
    X_train, X_test, y_train, y_test, preprocessor = build_preprocessor(df, target)

    # ---------------------------
    # 3) TRAIN MULTIPLE MODELS
    # ---------------------------
    trained_models, model_metrics, best_model_name = train_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    best_model = trained_models[best_model_name]

    # ---------------------------
    # 4) DRIFT MONITORING
    # ---------------------------
    drift_report = ks_drift(X_train, X_test)

    # ---------------------------
    # 5) SHAP EXPLAINABILITY
    # ---------------------------
    shap_importance = compute_shap(best_model, X_test)

    # ---------------------------
    # 6) CALIBRATION
    # ---------------------------
    calibration = calibration_analysis(
        y_test,
        best_model.predict_proba(X_test)[:, 1]
    )

    # ---------------------------
    # 7) BUILD CENTRAL PAYLOAD
    # ---------------------------
    payload = {
        "schema": schema,
        "model_metrics": model_metrics,
        "best_model": best_model_name,
        "drift": drift_report,
        "shap_importance": shap_importance,
        "calibration": calibration
    }

    # ---------------------------
    # 8) ALERTS (AUTOMATIC WARNINGS)
    # ---------------------------
    alerts = generate_alerts(drift_report, model_metrics)
    payload["alerts"] = alerts

    # ---------------------------
    # 9) SAVE ARTIFACTS (MLOPS PART)
    # ---------------------------
    save_json("audit_payload", payload)

    log_experiment(
        {"dataset_url": csv_source, "target": target},
        best_model_name,
        model_metrics
    )

    # ---------------------------
    # 10) GENAI REPORT (CHERRY ON TOP)
    # ---------------------------
    ai_report = generate_reasoned_audit(payload)
    save_json("ai_audit_report", {"report": ai_report})

    payload["ai_report"] = ai_report

    return payload
