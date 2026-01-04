# src/main.py
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.config import TSConfig
from src.data.io import load_flows_csv
from src.data.features import aggregate_flows_to_timeseries
from src.data.dataset import FlowWindowDataset, collate_fn
from src.modeling.model import build_model
from src.modeling.train import train_model
from src.modeling.score import forecast_and_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--window", default="5min")
    parser.add_argument("--context_length", type=int, default=48)
    parser.add_argument("--prediction_length", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--anomaly_threshold", type=float, default=3.0)
    parser.add_argument("--persistence_windows", type=int, default=3)
    args = parser.parse_args()

    cfg = TSConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # 1) Load raw flows
    df = load_flows_csv(args.csv)

    # 2) Aggregate to time-series windows
    agg = aggregate_flows_to_timeseries(df, window=args.window)

    feature_cols = [
        "flow_count",
        "bytes_sum",
        "packets_sum",
        "avg_duration",
        "unique_dst_ports",
        "tcp_ratio",
        "udp_ratio",
    ]
    time_feat_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "age"]

    group_series = {gid: gdf for gid, gdf in agg.groupby("group_id")}

    # 3) Fit scaler across all groups (in production: fit on a "clean" period)
    all_X = np.vstack(
        [gdf[feature_cols].to_numpy(dtype=np.float32) for gdf in group_series.values() if len(gdf) > 0]
    )
    scaler = StandardScaler().fit(all_X)

    # 4) Build model
    model = build_model(input_size=len(feature_cols), time_feat_dim=len(time_feat_cols), cfg=cfg)

    max_lag = max(model.config.lags_sequence) if model.config.lags_sequence else 0
    print("Using lags_sequence:", model.config.lags_sequence, "max_lag:", max_lag)

    # 5) Build datasets with max_lag so HF has enough past history
    train_ds = FlowWindowDataset(
        group_series=group_series,
        feature_cols=feature_cols,
        time_feat_cols=time_feat_cols,
        cfg=cfg,
        scaler=scaler,
        max_lag=max_lag,
        mode="train",
    )
    if len(train_ds) == 0:
        raise RuntimeError(
            "Not enough data to create training windows. "
            "Try reducing context_length/prediction_length or increasing data duration."
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Debug: confirm shapes
    b = next(iter(train_loader))
    print("past_values shape:", tuple(b["past_values"].shape))      # (B, context+max_lag, F)
    print("future_values shape:", tuple(b["future_values"].shape))  # (B, pred_len, F)

    # 6) Train
    train_model(model, train_loader, cfg)

    # 7) Score recent windows
    score_ds = FlowWindowDataset(
        group_series=group_series,
        feature_cols=feature_cols,
        time_feat_cols=time_feat_cols,
        cfg=cfg,
        scaler=scaler,
        max_lag=max_lag,
        mode="score",
    )

    alerts = forecast_and_score(
        model=model,
        dataset=score_ds,
        cfg=cfg,
        anomaly_threshold=args.anomaly_threshold,
        persistence_windows=args.persistence_windows,
    )

    flagged = alerts[alerts["is_alert"]].sort_values(["score"], ascending=False)
    print("\n=== Alerts (Potential Low-and-Slow C2) ===")
    if len(flagged) == 0:
        print("No alerts.")
    else:
        print(flagged.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
