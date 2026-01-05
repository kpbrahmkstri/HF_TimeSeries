# src/modeling/score.py
import numpy as np
import pandas as pd
import torch

from src.data.dataset import FlowWindowDataset
from src.config import TSConfig


@torch.no_grad()
def forecast_and_score(
    model,
    dataset: FlowWindowDataset,
    cfg: TSConfig,
    anomaly_threshold: float = 0.35,
    persistence_windows: int = 2,
) -> pd.DataFrame:
    """
    Scores each sliding window by comparing observed future vs probabilistic forecast:
        score = mean(|y_true - mean(y_samples)| / std(y_samples))

    Returns a DataFrame with:
      group_id, start_idx, pred_start_idx, start_time, score, n_samples, is_alert
    """
    model.eval()
    model.to(cfg.device)

    # Past length = context_length + max_lag (dataset sets this)
    past_len = getattr(dataset, "past_length", cfg.context_length)
    results = []

    for i in range(len(dataset)):
        gid, start = dataset.samples[i]
        item = dataset[i]
        batch = {k: item[k].unsqueeze(0).to(cfg.device) for k in item.keys()}

        gen = model.generate(
            past_values=batch["past_values"],
            past_time_features=batch["past_time_features"],
            past_observed_mask=batch["past_observed_mask"],
            static_categorical_features=batch["static_categorical_features"],
            static_real_features=batch["static_real_features"],
            future_time_features=batch["future_time_features"],
        )

        seq = gen.sequences
        # Expected: (B, num_samples, pred_len, input_size) OR (B, pred_len, input_size)
        if seq.dim() == 3:
            seq = seq.unsqueeze(1)

        y_samples = seq.detach().cpu().numpy()[0]  # (S, pred_len, F)
        if y_samples.ndim == 2:
            y_samples = y_samples[None, :, :]

        y_true = batch["future_values"].detach().cpu().numpy()[0]  # (pred_len, F)
        y_mean = y_samples.mean(axis=0)
        y_std = y_samples.std(axis=0) + 1e-6

        z = np.abs(y_true - y_mean) / y_std
        score = float(z.mean())

        # Always compute pred_start_idx
        pred_start_idx = int(start + past_len)

        # Try to map to timestamp if dataset stored TS
        start_time = None
        try:
            ts_arr = dataset.data.get(gid, {}).get("TS", None)
            if ts_arr is not None and pred_start_idx < len(ts_arr):
                start_time = ts_arr[pred_start_idx]
        except Exception:
            start_time = None

        # IMPORTANT: always include pred_start_idx and start_time keys
        results.append(
            {
                "group_id": gid,
                "start_idx": int(start),
                "pred_start_idx": pred_start_idx,
                "start_time": start_time,
                "score": score,
                "n_samples": int(y_samples.shape[0]),
            }
        )

    res = pd.DataFrame(results)

    # If nothing scored, return empty with expected columns
    if len(res) == 0:
        return pd.DataFrame(
            columns=["group_id", "start_idx", "pred_start_idx", "start_time", "score", "n_samples", "is_alert"]
        )

    res = res.sort_values(["group_id", "start_idx"]).reset_index(drop=True)

    # Persistence-based alerting
    res["is_high"] = res["score"] >= anomaly_threshold
    res["high_run"] = (
        res.groupby("group_id")["is_high"]
        .apply(lambda s: s.rolling(persistence_windows, min_periods=persistence_windows).sum())
        .reset_index(level=0, drop=True)
    )
    res["is_alert"] = res["high_run"].fillna(0) >= persistence_windows

    return res[["group_id", "start_idx", "pred_start_idx", "start_time", "score", "n_samples", "is_alert"]]
