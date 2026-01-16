import numpy as np
import pandas as pd

def analyze_schema(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=np.number).columns.tolist(),
        "missing_ratio": df.isna().mean().to_dict()
    }
