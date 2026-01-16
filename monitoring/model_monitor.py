def monitor_performance(train_metrics, test_metrics):
    alerts = {}

    for model in train_metrics:
        drop = train_metrics[model]["f1"] - test_metrics[model]["f1"]
        alerts[model] = {
            "f1_drop": float(drop),
            "risk_flag": bool(drop > 0.05)
        }

    return alerts
