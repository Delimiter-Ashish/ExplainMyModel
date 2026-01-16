import pandas as pd
from urllib.parse import urlparse

def load_csv(source: str) -> pd.DataFrame:
    """
    Load CSV from local path or URL
    """
    if source.startswith("http"):
        return pd.read_csv(source)
    return pd.read_csv(source)
