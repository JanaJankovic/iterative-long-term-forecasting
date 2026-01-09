from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any


import pandas as pd


from src.model.operations.train import fit_tabular_autoencoder
from src.utils.data_class import DataConfig, ModelConfig, TrainConfig
from src.data.preprocess import process_tabular_readout, process_tabular_transition
from src.model.architecture.autoencoder import TabularAutoencoder
from src.model.architecture.lantent import TabularLatentTransition, TabularLatentReadout
from src.model.operations.evaluate import compute_metrics
from src.utils.data_utils import json_dumps
import numpy as np
import datetime


def latent_transition_pipeline(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str,
    out_dir: str,
    verbose=False,
):
    loaders, scaler, input_shape, meta = process_tabular_transition(data_cfg)
    train_loader, val_loader, _ = loaders
    autoencoder = TabularAutoencoder(
        input_dim=input_shape,
        latent_dim=model_cfg.latent_dim,
        hidden_dim=model_cfg.hidden,
        num_layers=model_cfg.layers,
        activation=model_cfg.activation,
        dropout=model_cfg.dropout,
        variational=model_cfg.variational,
    )
    latent_regressor = TabularLatentTransition(autoencoder, model_cfg.latent_cfg)
    fit_tabular_autoencoder(
        autoencoder, train_loader, val_loader, device, model_cfg, train_cfg, verbose
    )

    X_all_s = meta["X_all_scaled"]
    X_all = meta["X_all"]
    n_train = meta["n_train"]
    horizon = data_cfg.horizon

    latent_meta = latent_regressor.forecast(X_all_s, n_train, horizon, device)
    X_hat_s = latent_meta["X_hat"]
    X_hat = scaler.inverse_transform(X_hat_s)

    y_pred = X_hat[:, 0].reshape(-1)
    y_true = X_all[n_train : n_train + horizon, 0].reshape(-1)
    metrics = compute_metrics(y_true, y_pred)

    tag = Path(data_cfg.csv_path).stem
    model_tag = f"{tag}_tabular_ld{int(model_cfg.latent_dim)}_{model_cfg.latent_cfg.regressor_name}"
    base = Path(out_dir) / "models" / model_tag

    # AE weights
    # torch.save(autoencoder.state_dict(), base.with_suffix(".ae.pt"))
    # scaler
    # joblib.dump(scaler, base.with_suffix(".scaler.pkl"))
    # latent forecaster (sklearn)
    # joblib.dump(latent_regressor.forecaster, base.with_suffix(".latent.pkl"))
    tz = datetime.timezone.utc
    dt = datetime.datetime.now(tz=tz).strftime("%Y%m%d%H%M%S")
    report = {
        "data_cfg": (
            asdict(data_cfg)
            if hasattr(data_cfg, "__dataclass_fields__")
            else dict(data_cfg=data_cfg)
        ),
        "device": device,
        "model_cfg": (
            asdict(model_cfg)
            if hasattr(model_cfg, "__dataclass_fields__")
            else dict(model_cfg=model_cfg)
        ),
        "train_cfg": (
            asdict(train_cfg)
            if hasattr(train_cfg, "__dataclass_fields__")
            else dict(train_cfg=train_cfg)
        ),
        "metrics": metrics,
        "X_all": np.array(meta["X_all"][:, 0]).tolist(),
        "y_pred": y_pred.tolist(),
    }
    (Path(out_dir) / "reports" / f"{dt}_{model_tag}.json").write_text(
        json_dumps(report), encoding="utf-8"
    )

    return {
        "name": f"{dt}_{model_tag}",
        "n_train": n_train,
        "effective_horizon": data_cfg.horizon,
        "X_all": meta["X_all"],
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "artifacts_base": str(base),
    }


def latent_readout_pipeline(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str,
    out_dir: str,
    verbose=False,
):

    loaders, input_shape, meta = process_tabular_readout(data_cfg)
    train_loader, val_loader, _ = loaders
    autoencoder = TabularAutoencoder(
        input_dim=input_shape,
        latent_dim=model_cfg.latent_dim,
        hidden_dim=model_cfg.hidden,
        num_layers=model_cfg.layers,
        activation=model_cfg.activation,
        dropout=model_cfg.dropout,
        variational=model_cfg.variational,
    )
    fit_tabular_autoencoder(
        autoencoder, train_loader, val_loader, device, model_cfg, train_cfg, verbose
    )

    regressor = TabularLatentReadout(
        ae_model=autoencoder,
        data_cfg=data_cfg,
        cfg=model_cfg.latent_cfg,
        lags=data_cfg.lags,
        rolling_windows=data_cfg.rolling_windows,
        calendar_cols=data_cfg.calendar_cols,  # e.g. ["dayofweek","month","dayofyear"]
        numeric_cols=data_cfg.numeric_cols,  # e.g. ["is_weekend","lag_1",...,"roll_mean_7",...]
        nominal_cols=data_cfg.nominal_cols,
        y_col="y",
        feature_names_in=meta["feature_names_in"],
        scalers=meta["scalers"],
        freq="D",
    )

    regressor.fit(meta["n_train"], device)
    forecast = regressor.forecast(meta["n_val_end"], data_cfg.horizon, device)

    n_train = meta["n_train"]
    y_pred = forecast["y_pred"]
    y_true = meta["y_all_unscaled"][n_train : n_train + data_cfg.horizon]
    metrics = compute_metrics(y_true, y_pred)

    tag = Path(data_cfg.csv_path).stem
    model_tag = f"{tag}_tabular_ld{int(model_cfg.latent_dim)}_{model_cfg.latent_cfg.regressor_name}"
    base = Path(out_dir) / "models" / model_tag

    # AE weights
    # torch.save(autoencoder.state_dict(), base.with_suffix(".ae.pt"))
    # scaler
    # joblib.dump(scaler, base.with_suffix(".scaler.pkl"))
    # latent forecaster (sklearn)
    # joblib.dump(latent_regressor.forecaster, base.with_suffix(".latent.pkl"))
    tz = datetime.timezone.utc
    dt = datetime.datetime.now(tz=tz).strftime("%Y%m%d%H%M%S")
    report = {
        "data_cfg": (
            asdict(data_cfg)
            if hasattr(data_cfg, "__dataclass_fields__")
            else dict(data_cfg=data_cfg)
        ),
        "device": device,
        "model_cfg": (
            asdict(model_cfg)
            if hasattr(model_cfg, "__dataclass_fields__")
            else dict(model_cfg=model_cfg)
        ),
        "train_cfg": (
            asdict(train_cfg)
            if hasattr(train_cfg, "__dataclass_fields__")
            else dict(train_cfg=train_cfg)
        ),
        "metrics": metrics,
        "X_all": np.array(meta["y_all_unscaled"]).tolist(),
        "y_pred": y_pred.tolist(),
    }
    (Path(out_dir) / "reports" / f"{dt}_{model_tag}.json").write_text(
        json_dumps(report), encoding="utf-8"
    )

    return {
        "name": f"{dt}_{model_tag}",
        "n_train": n_train,
        "effective_horizon": data_cfg.horizon,
        "X_all": meta["y_all_unscaled"],
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "artifacts_base": str(base),
    }
