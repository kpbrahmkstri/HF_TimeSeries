# src/modeling/model.py
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from src.config import TSConfig


def build_model(input_size: int, time_feat_dim: int, cfg: TSConfig) -> TimeSeriesTransformerForPrediction:
    """
    Build HF TimeSeriesTransformerForPrediction.

    NOTE:
    HF uses lagged features internally, requiring past_values length = context_length + max_lag.
    We will ensure lags are safe relative to prediction_length as well.
    """
    # Robust rule: avoid lag == prediction_length; keep lags strictly smaller.
    max_lag = max(1, cfg.prediction_length - 1)

    base_lags = [1, 2, 3]  # safe defaults; add 6 only if you also extend past_length accordingly
    lags = [l for l in base_lags if l <= max_lag]
    if not lags:
        lags = [1]

    model_cfg = TimeSeriesTransformerConfig(
        prediction_length=cfg.prediction_length,
        context_length=cfg.context_length,
        input_size=input_size,
        num_time_features=time_feat_dim,
        num_static_categorical_features=1,
        cardinality=[1],
        embedding_dimension=[2],
        num_static_real_features=1,
        lags_sequence=lags,
        num_parallel_samples=cfg.num_samples_forecast,
        d_model=64,
        encoder_layers=3,
        decoder_layers=3,
        dropout=0.1,
    )
    return TimeSeriesTransformerForPrediction(model_cfg)
