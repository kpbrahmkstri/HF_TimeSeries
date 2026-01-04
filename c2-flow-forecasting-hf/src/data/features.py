import numpy as np
import pandas as pd
from typing import Tuple

def _ensure_datetime_utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce")

def aggregate_flows_to_timeseries(
    df: pd.DataFrame,
    window: str = "5min",
    group_key: Tuple[str, str] = ("src_ip", "dst_ip"),
) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = _ensure_datetime_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    for c in ["bytes", "packets"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce").fillna(-1).astype(int)

    df["t_bucket"] = df["timestamp"].dt.floor(window)
    df["group_id"] = df[group_key[0]].astype(str) + "->" + df[group_key[1]].astype(str)

    agg = df.groupby(["group_id", "t_bucket"]).agg(
        flow_count=("protocol", "size"),
        bytes_sum=("bytes", "sum"),
        packets_sum=("packets", "sum"),
        avg_duration=("duration", "mean"),
        unique_dst_ports=("dst_port", "nunique"),
        tcp_ratio=("protocol", lambda x: float((x == "TCP").sum()) / max(len(x), 1)),
        udp_ratio=("protocol", lambda x: float((x == "UDP").sum()) / max(len(x), 1)),
    ).reset_index().rename(columns={"t_bucket": "timestamp"})

    ts = agg["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek.astype(float)

    agg["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    agg["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    agg["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    agg["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    agg["age"] = agg.groupby("group_id").cumcount()
    max_age = agg.groupby("group_id")["age"].transform("max").replace(0, 1)
    agg["age"] = agg["age"] / max_age

    return agg
