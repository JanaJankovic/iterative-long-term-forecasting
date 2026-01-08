from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from src.utils.constants import (
    RegressorName,
)


@dataclass
class DataConfig:
    csv_path: str
    target_col: str
    datetime_col: str
    batch_size: int
    split_ratio: Tuple[float, float, float]
    lags: Tuple[int, ...] = (1, 2, 7, 14, 30)
    rolling_windows: Tuple[int, ...] = (3, 7)
    datetime_col: str
    horizon: int = 1
    lookback: int = 7
    stride: int = 1


@dataclass
class OptimConfig:
    name: str = "adamw"  # "adam" | "adamw" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class LatentForecasterConfig:
    regressor_name: RegressorName = "random_forest"
    regressor_params: Dict[str, Any] = None  # set in __post_init__
    use_mu_not_sample: bool = True  # if VAE, use mu for stability

    def __post_init__(self):
        if self.regressor_params is None:
            self.regressor_params = dict(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
            )


@dataclass
class ModelConfig:
    # Data
    latent_dim: int
    horizon: int
    latent_cfg: LatentForecasterConfig = LatentForecasterConfig()
    optim: OptimConfig = OptimConfig()

    variational: bool = True
    beta_kl: float = 1e-3

    hidden: int = 128
    layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    encoder_bidirectional: bool = False


@dataclass
class TrainConfig:
    epochs_ae: int = 30
    loss_fn: str = "mse"  # "mse" or "mae" depending on your model implementation
    grad_clip: Optional[float] = 1.0
    early_stop_patience: int = 7
