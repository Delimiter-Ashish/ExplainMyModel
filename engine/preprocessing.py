import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df: pd.DataFrame):
    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    df = df.dropna()
    return df, encoders
