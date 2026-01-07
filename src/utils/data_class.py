from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from src.utils.constants import (
    RegressorName,
    EncoderType,
    DecoderType,
    LatentFeatureMode,
)


@dataclass
class OptimConfig:
    name: str = "adamw"  # "adam" | "adamw" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class ModelConfig:
    # Data
    d_in: int
    latent_dim: int
    horizon: int

    # Structured latent
    variational: bool = True
    beta_kl: float = 1e-3

    # Encoder / Decoder selection
    encoder_type: EncoderType = "rnn"
    decoder_type: DecoderType = "mlp"

    # MLP params
    mlp_hidden: int = 128
    mlp_layers: int = 2
    activation: str = "relu"
    mlp_dropout: float = 0.0

    # RNN params
    rnn_hidden: int = 128
    rnn_layers: int = 2
    rnn_type: str = "gru"
    rnn_dropout: float = 0.0
    encoder_bidirectional: bool = False

    # Latent regressor config (the "simple predictor")
    regressor_name: RegressorName = "ridge"
    regressor_params: Dict[str, Any] = field(default_factory=dict)
    latent_feature_mode: LatentFeatureMode = "last"  # how z_past -> features

    # Torch optimizer (for encoder+decoder training only)
    optim: OptimConfig = OptimConfig()


@dataclass
class IterForecastResult:
    # scaled space
    seed_x_scaled: np.ndarray  # (T, 1)
    y_pred_scaled: np.ndarray  # (S, 1)

    # original space (if scaler provided)
    seed_x: Optional[np.ndarray]  # (T, 1)
    y_pred: Optional[np.ndarray]  # (S, 1)


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
