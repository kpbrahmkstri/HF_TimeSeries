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
    Windowed dataset for Hugging Face TimeSeriesTransformer.

    Key requirement:
      HF TimeSeriesTransformer uses lagged features internally.
      Therefore, past_values must include extra history:
        past_length = context_length + max_lag

    Each sample returns:
      past_values:          (past_length, F)
      future_values:        (prediction_length, F)
      past_time_features:   (past_length, T)
      future_time_features: (prediction_length, T)
      masks + static placeholders

    Additionally stores per-group timestamps (TS) so scoring can map windows to real time.
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

        # Stores sliding window start indices: (group_id, start_idx)
        self.samples: List[Tuple[str, int]] = []

        # Store per-group arrays to speed up indexing
        # X: scaled features, TF: time features, TS: timestamps
        self.data: Dict[str, Dict[str, np.ndarray]] = {}

        for gid, gdf in group_series.items():
            if gdf is None or len(gdf) == 0:
                continue

            gdf = gdf.sort_values("timestamp").reset_index(drop=True)

            # Main features (scaled)
            X = gdf[self.feature_cols].to_numpy(dtype=np.float32)
            Xs = self.scaler.transform(X).astype(np.float32)

            # Time features (not scaled)
            TF = gdf[self.time_feat_cols].to_numpy(dtype=np.float32)

            # Timestamps (for mapping windows -> time in scoring/plots)
            TS = gdf["timestamp"].to_numpy()

            self.data[gid] = {"X": Xs, "TF": TF, "TS": TS}

            total_len = len(gdf)
            if total_len < self.total_length:
                continue

            if mode == "train":
                # All windows (sliding)
                for start in range(0, total_len - self.total_length + 1, cfg.stride):
                    self.samples.append((gid, start))
            else:
                # For scoring: keep only the most recent portion (speeds up scoring)
                start_min = max(0, total_len - 5 * self.total_length)
                for start in range(start_min, total_len - self.total_length + 1, cfg.stride):
                    self.samples.append((gid, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gid, start = self.samples[idx]
        X = self.data[gid]["X"]
        TF = self.data[gid]["TF"]

        pl = self.past_length
        p = self.cfg.prediction_length

        # Provide extra history for lag extraction
        past_values = X[start : start + pl]                    # (pl, F)
        future_values = X[start + pl : start + pl + p]         # (p, F)

        past_time_features = TF[start : start + pl]            # (pl, T)
        future_time_features = TF[start + pl : start + pl + p] # (p, T)

        # Masks: all observed
        past_observed_mask = np.ones_like(past_values, dtype=np.float32)
        future_observed_mask = np.ones_like(future_values, dtype=np.float32)

        # Static placeholders required by HF signature
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


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack tensors into a batch."""
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}
