# src/data/dataset.py
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from src.config import TSConfig


class FlowWindowDataset(Dataset):
    """
    Produces model-ready dicts for HF TimeSeriesTransformer.

    IMPORTANT:
    HF TimeSeriesTransformer requires extra history to compute lagged features.
    We must provide: past_values length = context_length + max_lag
    """

    def __init__(
        self,
        group_series: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        time_feat_cols: List[str],
        cfg: TSConfig,
        scaler: StandardScaler,
        max_lag: int,
        mode: str = "train",
    ):
        self.cfg = cfg
        self.feature_cols = feature_cols
        self.time_feat_cols = time_feat_cols
        self.scaler = scaler
        self.mode = mode

        self.max_lag = int(max_lag)
        self.past_length = cfg.context_length + self.max_lag
        self.total_length = self.past_length + cfg.prediction_length

        self.samples: List[Tuple[str, int]] = []
        self.data: Dict[str, Dict[str, np.ndarray]] = {}

        for gid, gdf in group_series.items():
            gdf = gdf.sort_values("timestamp").reset_index(drop=True)

            X = gdf[feature_cols].to_numpy(dtype=np.float32)
            TF = gdf[time_feat_cols].to_numpy(dtype=np.float32)

            # Scale only the main features (not time features)
            Xs = scaler.transform(X).astype(np.float32)

            self.data[gid] = {"X": Xs, "TF": TF}

            total_len = len(gdf)
            if total_len < self.total_length:
                continue

            if mode == "train":
                for start in range(0, total_len - self.total_length + 1, cfg.stride):
                    self.samples.append((gid, start))
            else:
                # Keep a few most recent windows for scoring
                start_min = max(0, total_len - 5 * self.total_length)
                for start in range(start_min, total_len - self.total_length + 1, cfg.stride):
                    self.samples.append((gid, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        gid, start = self.samples[idx]
        X = self.data[gid]["X"]
        TF = self.data[gid]["TF"]

        p = self.cfg.prediction_length
        pl = self.past_length

        # Provide extra history for lag extraction
        past_values = X[start : start + pl]                     # (context+max_lag, F)
        future_values = X[start + pl : start + pl + p]          # (p, F)

        past_time_features = TF[start : start + pl]             # (context+max_lag, TFD)
        future_time_features = TF[start + pl : start + pl + p]  # (p, TFD)

        # Observed masks (set 0 where missing if you add missingness later)
        past_observed_mask = np.ones_like(past_values, dtype=np.float32)
        future_observed_mask = np.ones_like(future_values, dtype=np.float32)

        # Static features placeholders (required by HF signature)
        static_categorical_features = np.zeros((1,), dtype=np.int64)
        static_real_features = np.zeros((1,), dtype=np.float32)

        return {
            "past_values": torch.tensor(past_values),
            "future_values": torch.tensor(future_values),
            "past_time_features": torch.tensor(past_time_features),
            "future_time_features": torch.tensor(future_time_features),
            "past_observed_mask": torch.tensor(past_observed_mask),
            "future_observed_mask": torch.tensor(future_observed_mask),
            "static_categorical_features": torch.tensor(static_categorical_features),
            "static_real_features": torch.tensor(static_real_features),
        }


def collate_fn(batch: List[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}
