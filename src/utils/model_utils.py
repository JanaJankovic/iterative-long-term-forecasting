from typing import Any, Dict
import torch
import torch.nn as nn
import random
import numpy as np

from src.utils.constants import RegressorName
from src.utils.data_class import OptimConfig
import torch
import torch.nn as nn
import inspect


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


def _filter_kwargs_for_ctor(ctor, params: Dict[str, Any]) -> Dict[str, Any]:
    if params is None:
        return {}
    sig = inspect.signature(ctor)
    accepted = set(sig.parameters.keys())
    accepted.discard("self")

    # If ctor accepts **kwargs, pass everything through
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(params)

    # Otherwise drop unknown keys
    return {k: v for k, v in params.items() if k in accepted}


def build_regressor(name: RegressorName, params: Dict[str, Any]) -> Any:
    name = name.lower()
    params = {} if params is None else dict(params)

    if name == "linear":
        from sklearn.linear_model import LinearRegression

        ctor = LinearRegression

    elif name == "ridge":
        from sklearn.linear_model import Ridge

        ctor = Ridge

    elif name == "lasso":
        from sklearn.linear_model import Lasso

        ctor = Lasso

    elif name == "elasticnet":
        from sklearn.linear_model import ElasticNet

        ctor = ElasticNet

    elif name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        ctor = RandomForestRegressor

    elif name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        ctor = ExtraTreesRegressor

    elif name == "xgboost":
        from xgboost import XGBRegressor

        ctor = XGBRegressor

    else:
        raise ValueError(f"Unknown regressor_name: {name}")

    used = _filter_kwargs_for_ctor(ctor, params)
    return ctor(**used)


def configure_optimizer(cfg: OptimConfig, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg.name.lower()
    lr = float(cfg.lr)
    wd = float(cfg.weight_decay)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {cfg.name}")
