import csv
import os
from datetime import datetime

LOG_PATH = "artifacts/experiment_log.csv"

def log_experiment(config, best_model, metrics):
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "dataset",
                "target",
                "best_model",
                "best_f1"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            config.get("dataset_url"),
            config.get("target"),
            best_model,
            metrics[best_model]["f1"]
        ])

