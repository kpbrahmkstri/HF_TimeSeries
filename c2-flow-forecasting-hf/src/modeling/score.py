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
    anomaly_threshold: float = 3.0,
    persistence_windows: int = 3,
) -> pd.DataFrame:
    """
    Forecast future windows and compute anomaly score using forecast uncertainty.

    This version is robust across transformers versions by NOT passing `num_samples`
    into model.generate(). Sampling count is controlled via model.config.num_parallel_samples.
    """
    model.eval()
    model.to(cfg.device)

    # Confirm how many samples the model will generate (probabilistic forecasts)
    n_samples = getattr(model.config, "num_parallel_samples", 1)

    results = []
    for i in range(len(dataset)):
        gid, start = dataset.samples[i]
        item = dataset[i]
        batch = {k: item[k].unsqueeze(0).to(cfg.device) for k in item.keys()}

        # IMPORTANT: do NOT pass num_samples here; some transformers versions don't accept it.
        gen = model.generate(
            past_values=batch["past_values"],
            past_time_features=batch["past_time_features"],
            past_observed_mask=batch["past_observed_mask"],
            static_categorical_features=batch["static_categorical_features"],
            static_real_features=batch["static_real_features"],
            future_time_features=batch["future_time_features"],
        )

        seq = gen.sequences

        # Handle shapes across versions:
        # expected: (batch, num_samples, pred_len, input_size)
        if seq.dim() == 3:
            # (batch, pred_len, input_size) or (batch, num_samples, pred_len) for univariate
            # Try to interpret: if last dim equals input_size -> treat as deterministic forecast (1 sample)
            # Otherwise add last dim.
            seq = seq.unsqueeze(1)

        # Ensure we have: (num_samples, pred_len, input_size)
        y_samples = seq.detach().cpu().numpy()[0]  # (S, P, F) hopefully
        if y_samples.ndim == 2:
            # (P, F) -> add sample dim
            y_samples = y_samples[None, :, :]

        # Observed future (scaled)
        y_true = batch["future_values"].detach().cpu().numpy()[0]  # (P, F)

        # Mean/std across samples (probabilistic)
        y_mean = y_samples.mean(axis=0)            # (P, F)
        y_std = y_samples.std(axis=0) + 1e-6       # (P, F)

        # Normalized absolute error
        z = np.abs(y_true - y_mean) / y_std        # (P, F)

        # Score: mean z across time + features
        score = float(z.mean())

        results.append({
            "group_id": gid,
            "start_idx": start,
            "score": score,
            "n_samples": int(y_samples.shape[0]),
        })

    res = pd.DataFrame(results).sort_values(["group_id", "start_idx"]).reset_index(drop=True)

    # Persistence logic: alert only if high scores persist across consecutive windows
    res["is_high"] = res["score"] >= anomaly_threshold
    res["high_run"] = (
        res.groupby("group_id")["is_high"]
        .apply(lambda s: s.rolling(persistence_windows).sum())
        .reset_index(level=0, drop=True)
    )
    res["is_alert"] = res["high_run"] >= persistence_windows

    return res[["group_id", "start_idx", "score", "n_samples", "is_alert"]]
