import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from src.utils.constants import LatentFeatureMode


def latent_to_features(z_past: torch.Tensor, mode: LatentFeatureMode) -> np.ndarray:
    """
    z_past: (B, T, L) torch -> features: (B, F) numpy
    """
    if z_past.ndim != 3:
        raise ValueError("z_past must be (B, T, L)")
    if mode == "last":
        feats = z_past[:, -1, :]  # (B, L)
    elif mode == "mean":
        feats = z_past.mean(dim=1)  # (B, L)
    elif mode == "flatten":
        feats = z_past.reshape(z_past.shape[0], -1)  # (B, T*L)
    else:
        raise ValueError(f"Unknown latent_feature_mode: {mode}")

    return feats.detach().cpu().float().numpy()


def latent_future_to_targets(z_future: torch.Tensor) -> np.ndarray:
    """
    z_future: (B, H, L) -> y: (B, H*L) numpy
    """
    if z_future.ndim != 3:
        raise ValueError("z_future must be (B, H, L)")
    B, H, L = z_future.shape
    return z_future.detach().cpu().float().numpy().reshape(B, H * L)


def collect_tensors_from_loader(
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collects all batches into two tensors:
      x_all: (N, T, 1)
      y_all: (N, H, 1)
    """
    xs = []
    ys = []
    for x, y in loader:
        xs.append(x.to(device))
        ys.append(y.to(device).unsqueeze(-1))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def cyclic_encode_calendar_features(
    df: pd.DataFrame,
    calendar_cols: List[str],
    *,
    drop_original: bool = True,
) -> pd.DataFrame:
    cycle_map = {
        "dayofweek": (7, 0.0),
        "dow": (7, 0.0),
        "weekday": (7, 0.0),
        "month": (12, 1.0),
        "dayofyear": (366, 1.0),  # safe across leap years
        "doy": (366, 1.0),
        "weekofyear": (53, 1.0),  # safer than 52
        "woy": (53, 1.0),
        "hour": (24, 0.0),
        "minute": (60, 0.0),
        "second": (60, 0.0),
    }

    out = df.copy()
    for col in calendar_cols:
        if col not in out.columns:
            continue

        key = col.lower().replace("_", "")
        P, offset = cycle_map.get(key, (None, None))
        if P is None:
            raise ValueError(f"Unknown calendar column '{col}'. Add it to cycle_map.")

        v = out[col].astype(float).to_numpy()
        a = 2.0 * np.pi * ((v - offset) / float(P))
        out[f"{col}_sin"] = np.sin(a).astype(np.float32)
        out[f"{col}_cos"] = np.cos(a).astype(np.float32)

        if drop_original:
            out = out.drop(columns=[col])

    return out


def split_scale_encode_df(
    df: pd.DataFrame,
    *,
    calendar_cols: List[str],
    nominal_cols: List[str],
    numeric_cols: List[str],
    y_col: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    numeric_scaler=None,  # e.g., MinMaxScaler() or StandardScaler()
    y_scaler=None,  # if None -> use numeric_scaler
    onehot: Optional[OneHotEncoder] = None,
    drop_original_calendar: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    - calendar_cols: cyclic sin/cos encoding
    - nominal_cols: one-hot encoded (fit on train, transform val/test)
    - numeric_cols: scaled (fit on train only)
    - y_col: scaled with y_scaler if provided; else uses numeric_scaler
    """
    df = cyclic_encode_calendar_features(
        df, calendar_cols, drop_original=drop_original_calendar
    )
    df = df.copy()

    n = len(df)
    n_train = int(n * split_ratio[0])
    n_val_end = n_train + int(n * split_ratio[1])

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_val_end].copy()
    test = df.iloc[n_val_end:].copy()

    if numeric_scaler is None:
        numeric_scaler = MinMaxScaler()
    if onehot is None:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    used_y_scaler = y_scaler if y_scaler is not None else numeric_scaler

    # --- One-hot nominal ---
    if nominal_cols:
        # make sure the encoder does not crash on unseen categories
        if getattr(onehot, "handle_unknown", None) != "ignore":
            onehot.set_params(handle_unknown="ignore")

        onehot.fit(train[nominal_cols])

        tr_mat = onehot.transform(train[nominal_cols])
        va_mat = onehot.transform(val[nominal_cols])
        te_mat = onehot.transform(test[nominal_cols])

        # FIX: densify (works for scipy sparse)
        if hasattr(tr_mat, "toarray"):
            tr_mat = tr_mat.toarray()
            va_mat = va_mat.toarray()
            te_mat = te_mat.toarray()

        oh_cols = onehot.get_feature_names_out(nominal_cols)

        tr_oh = pd.DataFrame(tr_mat, index=train.index, columns=oh_cols)
        va_oh = pd.DataFrame(va_mat, index=val.index, columns=oh_cols)
        te_oh = pd.DataFrame(te_mat, index=test.index, columns=oh_cols)

        train = pd.concat([train.drop(columns=nominal_cols), tr_oh], axis=1)
        val = pd.concat([val.drop(columns=nominal_cols), va_oh], axis=1)
        test = pd.concat([test.drop(columns=nominal_cols), te_oh], axis=1)

    # --- Scale numeric (excluding y if present) ---
    num_cols_no_y = [
        c
        for c in df.columns
        if c != y_col and any(c.startswith(base) for base in numeric_cols)
    ]
    if num_cols_no_y:
        numeric_scaler.fit(train[num_cols_no_y])
        train.loc[:, num_cols_no_y] = numeric_scaler.transform(train[num_cols_no_y])
        val.loc[:, num_cols_no_y] = numeric_scaler.transform(val[num_cols_no_y])
        test.loc[:, num_cols_no_y] = numeric_scaler.transform(test[num_cols_no_y])

    # --- Scale y separately (or reuse numeric_scaler) ---
    if y_col:
        used_y_scaler.fit(train[[y_col]])
        train.loc[:, [y_col]] = used_y_scaler.transform(train[[y_col]])
        val.loc[:, [y_col]] = used_y_scaler.transform(val[[y_col]])
        test.loc[:, [y_col]] = used_y_scaler.transform(test[[y_col]])

    scalers = {
        "numeric_scaler": numeric_scaler,
        "y_scaler": used_y_scaler,
        "onehot": onehot,
    }
    return train, val, test, scalers
