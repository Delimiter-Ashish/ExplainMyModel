def generate_alerts(drift_report, perf_report):
    alerts = []

    for col, info in drift_report.items():
        if info["drift_flag"]:
            alerts.append(f"⚠️ High drift detected in {col}")

    for model, info in perf_report.items():
        if info["risk_flag"]:
            alerts.append(f"⚠️ Performance degradation in {model}")

    return alerts
