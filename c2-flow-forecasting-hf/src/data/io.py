import pandas as pd

REQUIRED_COLS = [
    "timestamp","src_ip","dst_ip","src_port","dst_port",
    "protocol","bytes","packets","duration","flags"
]

def load_flows_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(list(set(REQUIRED_COLS) - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
