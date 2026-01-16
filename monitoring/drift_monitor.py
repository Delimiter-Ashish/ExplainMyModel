import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def ks_drift(train_df, test_df):
    drift = {}

    for col in train_df.select_dtypes(include=["int64","float64"]).columns:
        stat, p_value = ks_2samp(train_df[col], test_df[col])
        drift[col] = {
            "ks_stat": float(stat),
            "p_value": float(p_value),
            "drift_flag": bool(p_value < 0.05)
        }

    return drift
