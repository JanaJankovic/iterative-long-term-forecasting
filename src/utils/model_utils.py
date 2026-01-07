from typing import Any, Dict
import torch
import torch.nn as nn
import random
import numpy as np

from src.utils.constants import RegressorName


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(0.01)
    raise ValueError(f"Unknown activation: {name}")


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_div_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL(q||p) with p=N(0,1), diagonal Gaussian
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_regressor(name: RegressorName, params: Dict[str, Any]):
    name = name.lower()

    if name == "linear":
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**params)

    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**params)

    if name == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso(**params)

    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(**params)

    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)

    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(**params)

    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params)

    raise ValueError(f"Unknown regressor_name: {name}")
