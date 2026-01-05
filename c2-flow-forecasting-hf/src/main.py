# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.config import TSConfig
from src.data.io import load_flows_csv
from src.data.features import aggregate_flows_to_timeseries
from src.data.dataset import FlowWindowDataset, collate_fn
from src.modeling.model import build_model
from src.modeling.train import train_model
from src.modeling.score import forecast_and_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to raw network-flow CSV.")
    parser.add_argument("--window", default="5min", help="Aggregation window (e.g., 1min, 5min).")

    parser.add_argument("--context_length", type=int, default=48)
    parser.add_argument("--prediction_length", type=int, default=12)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--anomaly_threshold", type=float, default=0.52)
    parser.add_argument("--persistence_windows", type=int, default=4)

    parser.add_argument(
        "--train_hours",
        type=float,
        default=12.0,
        help="Train only on first N hours (baseline).",
    )

    # Outputs / plotting
    parser.add_argument("--out_dir", default="results", help="Directory to write CSV + plots.")
    parser.add_argument("--plot_top_n", type=int, default=5, help="Plot top N group_ids by max score.")
    parser.add_argument("--c2_ip", default="198.51.100.77", help="C2 dst_ip to highlight/filter.")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = TSConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # 1) Load raw flows
    df = load_flows_csv(args.csv)

    # 2) Aggregate into time-series windows
    agg = aggregate_flows_to_timeseries(df, window=args.window)

    feature_cols = [
        "flow_count",
        "bytes_sum",
        "bytes_mean",
        "bytes_std",
        "packets_sum",
        "avg_duration",
        "duration_std",
        "unique_dst_ports",
        "tcp_ratio",
        "udp_ratio",
        "port_443_ratio",
        "dns_ratio",
        "small_flow_ratio",
    ]
    time_feat_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "age"]

    # Full series per group_id (used for scoring)
    group_series = {gid: gdf for gid, gdf in agg.groupby("group_id")}

    # 3) Baseline-only series for training
    t0 = agg["timestamp"].min()
    train_cutoff = t0 + np.timedelta64(int(args.train_hours * 3600), "s")

    train_group_series = {}
    for gid, gdf in group_series.items():
        train_part = gdf[gdf["timestamp"] < train_cutoff]
        if len(train_part) > 0:
            train_group_series[gid] = train_part

    if len(train_group_series) == 0:
        raise RuntimeError("train_group_series is empty. Increase --train_hours or check timestamps.")

    # 4) Fit scaler on baseline only
    all_X = np.vstack(
        [gdf[feature_cols].to_numpy(dtype=np.float32) for gdf in train_group_series.values() if len(gdf) > 0]
    )
    scaler = StandardScaler().fit(all_X)

    # 5) Build model
    model = build_model(input_size=len(feature_cols), time_feat_dim=len(time_feat_cols), cfg=cfg)
    max_lag = max(model.config.lags_sequence) if model.config.lags_sequence else 0
    print("Using lags_sequence:", model.config.lags_sequence, "max_lag:", max_lag)

    # 6) Train dataset/loader
    train_ds = FlowWindowDataset(
        group_series=train_group_series,
        feature_cols=feature_cols,
        time_feat_cols=time_feat_cols,
        cfg=cfg,
        scaler=scaler,
        max_lag=max_lag,
        mode="train",
    )
    if len(train_ds) == 0:
        raise RuntimeError(
            "Not enough baseline data to create training windows. "
            "Increase --train_hours or reduce --context_length/--prediction_length."
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Debug: confirm shapes
    b = next(iter(train_loader))
    print("past_values shape:", tuple(b["past_values"].shape))
    print("future_values shape:", tuple(b["future_values"].shape))

    # 7) Train
    train_model(model, train_loader, cfg)

    # 8) Score on FULL series
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

    # Make start_time plottable if present
    if "start_time" in alerts.columns:
        alerts["start_time"] = pd.to_datetime(alerts["start_time"], errors="coerce")

    # 9) Write CSV outputs
    top_scores = alerts.sort_values("score", ascending=False).head(100)
    top_scores_path = os.path.join(args.out_dir, "top_scores.csv")
    top_scores.to_csv(top_scores_path, index=False)

    flagged = alerts[alerts.get("is_alert", False)].sort_values("score", ascending=False) if "is_alert" in alerts.columns else alerts.iloc[0:0]
    alerts_path = os.path.join(args.out_dir, "alerts.csv")
    flagged.to_csv(alerts_path, index=False)

    c2_alerts = flagged[flagged["group_id"].str.contains(args.c2_ip, na=False)] if len(flagged) else flagged
    c2_alerts_path = os.path.join(args.out_dir, "c2_alerts.csv")
    c2_alerts.to_csv(c2_alerts_path, index=False)

    print(f"\nWrote: {top_scores_path}")
    print(f"Wrote: {alerts_path}")
    print(f"Wrote: {c2_alerts_path}")

    # 10) Plot anomaly scores over time for top N group_ids + C2
    by_max = alerts.groupby("group_id")["score"].max().sort_values(ascending=False)
    top_groups = list(by_max.head(args.plot_top_n).index)

    c2_groups = [g for g in alerts["group_id"].unique() if args.c2_ip in str(g)]
    plot_groups = []
    for g in top_groups:
        if g not in plot_groups:
            plot_groups.append(g)
    for g in c2_groups:
        if g not in plot_groups:
            plot_groups.append(g)

    # Choose best x-axis available (robust to older score.py without pred_start_idx)
    if "start_time" in alerts.columns and alerts["start_time"].notna().any():
        x_col = "start_time"
    elif "pred_start_idx" in alerts.columns:
        x_col = "pred_start_idx"
    else:
        x_col = "start_idx"

    plt.figure()
    for gid in plot_groups:
        gdf = alerts[alerts["group_id"] == gid].copy()
        gdf = gdf.sort_values(x_col)
        plt.plot(gdf[x_col], gdf["score"], label=gid)

    # threshold line
    plt.axhline(args.anomaly_threshold)

    plt.title("Anomaly score over time (HF TimeSeriesTransformer)")
    plt.xlabel(x_col)
    plt.ylabel("score")
    plt.xticks(rotation=30)
    plt.legend(fontsize="small")
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, "anomaly_scores.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Wrote: {plot_path}")

    # 11) Console debug prints
    print("\n=== Top anomaly scores (debug) ===")
    # Only print columns that exist (robust)
    print(top_scores.head(20).to_string(index=False))

    print(f"\n=== Scores for C2 dst_ip {args.c2_ip} (debug) ===")
    c2_scores_dbg = alerts[alerts["group_id"].str.contains(args.c2_ip, na=False)] \
        .sort_values("score", ascending=False).head(30)
    print(c2_scores_dbg.to_string(index=False) if len(c2_scores_dbg) else f"No rows for {args.c2_ip} in scoring set.")

    print("\n=== Alerts (Potential Low-and-Slow C2) ===")
    if len(flagged) == 0:
        print("No alerts.")
    else:
        print(flagged.head(50).to_string(index=False))

    print(f"\n=== C2 Alerts only ({args.c2_ip}) ===")
    if len(c2_alerts) == 0:
        print("No C2 alerts.")
    else:
        print(c2_alerts.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
