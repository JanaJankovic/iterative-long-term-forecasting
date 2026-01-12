from src.utils.data_class import ModelConfig, TrainConfig, DataConfig
from src.model.operations.pipeline import (
    latent_readout_pipeline,
    latent_transition_pipeline,
)
from dataclasses import replace
from typing import Any, Dict, List, Optional
import numpy as np


def random_search_tabular_latent(
    base_seed: int,
    n_trials: int,
    space: Dict[str, Any],
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    out_dir: str = "./../output",
    device: str = "cuda",
    metric_key: str = "rmse",  # must exist in metrics dict
    minimize: bool = True,
    verbose: bool = False,
    transitional: bool = True,
) -> Dict[str, Any]:
    """
    Random-search HPO wrapper around tabular_latent_pipeline.

    `space` format examples:
      space = {
        "split_ratio": [(0.6,0.2,0.2), (0.7,0.1,0.2)],
        "horizon": [24, 48],
        "lookback": [24, 72],
        "latent_dim": [4, 8, 16],
        "variational": [False, True],
        "beta_kl": [0.0, 1e-4, 1e-3],
        "latent_cfg.regressor_name": ["random_forest", "ridge"],
        "latent_cfg.regressor_params": [
            {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
            {"n_estimators": 600, "random_state": 42, "n_jobs": -1},
        ],
        "train_cfg.epochs_ae": [20, 40],
      }

    Returns dict with best + all trials.
    """
    rng = np.random.default_rng(base_seed)

    def pick(v):
        if isinstance(v, (list, tuple)):
            return v[int(rng.integers(0, len(v)))]
        raise TypeError("space values must be list/tuple options")

    def set_nested(dc: Any, key: str, value: Any) -> Any:
        # supports: "field", "latent_cfg.regressor_name", "train_cfg.epochs_ae"
        parts = key.split(".")
        if len(parts) == 1:
            return replace(dc, **{parts[0]: value})

        head, rest = parts[0], ".".join(parts[1:])
        sub = getattr(dc, head)
        sub2 = set_nested(sub, rest, value)  # recurse into dataclass
        return replace(dc, **{head: sub2})

    trials: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for i in range(n_trials):
        # deterministic per-trial seed for your run(s)
        s = int(rng.integers(0, 2**31 - 1))

        # sample a configuration
        dcfg = data_cfg
        mcfg = model_cfg
        tcfg = train_cfg
        sampled: Dict[str, Any] = {}

        for k, options in space.items():
            val = pick(options)
            sampled[k] = val
            if k.startswith("data_cfg."):
                dcfg = set_nested(dcfg, k[len("data_cfg.") :], val)
            elif k.startswith("model_cfg."):
                mcfg = set_nested(mcfg, k[len("model_cfg.") :], val)
            elif k.startswith("train_cfg."):
                tcfg = set_nested(tcfg, k[len("train_cfg.") :], val)
            else:
                # default routing by common names
                if hasattr(dcfg, k):
                    dcfg = set_nested(dcfg, k, val)
                elif hasattr(mcfg, k):
                    mcfg = set_nested(mcfg, k, val)
                elif hasattr(tcfg, k):
                    tcfg = set_nested(tcfg, k, val)
                else:
                    # allow explicit nested keys like "latent_cfg.regressor_name"
                    if k.startswith("latent_cfg."):
                        mcfg = set_nested(
                            mcfg, "latent_cfg." + k[len("latent_cfg.") :], val
                        )
                    else:
                        raise KeyError(f"Unknown hyperparameter key: {k}")

        if transitional:
            # run
            res = latent_transition_pipeline(
                dcfg, mcfg, tcfg, device=device, out_dir=out_dir, verbose=verbose
            )
            X_all = res["X_all"][:, 0]
        else:
            res = latent_readout_pipeline(
                dcfg, mcfg, tcfg, device=device, out_dir=out_dir, verbose=verbose
            )
            X_all = res["X_all"]
        metrics = res["metrics"]
        if metric_key not in metrics:
            raise KeyError(
                f"metric_key='{metric_key}' not found. metrics keys={list(metrics.keys())}"
            )

        score = float(metrics[metric_key])
        n_train = int(res.get("n_train", 0))
        n_val_end = int(res.get("n_val_end", 0))

        trial = {
            "trial": i,
            "name": res["name"],
            "seed": s,
            "score": score,
            "metrics": metrics,
            "X_all": np.array(X_all).tolist(),
            "y_pred": res["y_pred"].tolist(),
            "n_train": n_train,  # split index: train ends at n_train-1
            "n_val_end": n_val_end,  # split index: val ends at n_val_end-1 (test starts at n_val_end)
        }
        trials.append(trial)

        if best is None:
            best = trial
        else:
            better = (score < best["score"]) if minimize else (score > best["score"])
            if better:
                best = trial

    return {"best": best, "trials": trials}
