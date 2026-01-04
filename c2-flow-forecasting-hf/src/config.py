from dataclasses import dataclass
import torch

@dataclass
class TSConfig:
    context_length: int = 48
    prediction_length: int = 12
    stride: int = 1
    batch_size: int = 32
    epochs: int = 8
    lr: float = 2e-4
    num_samples_forecast: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
