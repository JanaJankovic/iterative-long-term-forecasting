from typing import Tuple
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data_class import DataConfig
from src.utils.data_utils import split_scale_encode_df


def build_sequence(data, lookback, horizon, stride=1):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1, stride):
        X.append(data[i - lookback : i])
        y.append(data[i : i + horizon])
    return np.array(X), np.array(y).squeeze(-1)


def process_sequences(data_cfg: DataConfig):
    df = pd.read_csv(data_cfg.csv_path)
    df = df[["load"]]

    # SPLIT DATA
    total_len = len(df)
    train_end = int(total_len * data_cfg.split_ratio[0])
    val_end = train_end + int(total_len * data_cfg.split_ratio[1])

    train_split = df.iloc[:train_end].copy()
    val_split = df.iloc[train_end:val_end].copy()
    test_split = df.iloc[val_end:].copy()

    # SCALE DATA
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_split)
    val_scaled = scaler.transform(val_split)
    test_scaled = scaler.transform(test_split)

    # BUILD SEQUENCES
    X_train, y_train = build_sequence(train_scaled, data_cfg.lookback, data_cfg.horizon)
    X_val, y_val = build_sequence(val_scaled, data_cfg.lookback, data_cfg.horizon)
    X_test, y_test = build_sequence(test_scaled, data_cfg.lookback, data_cfg.horizon)

    # CREATE DATA LOADERS
    X_train = torch.tensor(X_train, dtype=torch.float32)  # (N, L, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32)  # (N, H)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    loaders = (
        DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=data_cfg.batch_size,
            shuffle=False,
        ),
        DataLoader(
            TensorDataset(X_val, y_val), batch_size=data_cfg.batch_size, shuffle=False
        ),
        DataLoader(
            TensorDataset(X_test, y_test), batch_size=data_cfg.batch_size, shuffle=False
        ),
    )

    input_shape = X_train.shape[1:]

    return loaders, scaler, input_shape


def create_features_from_series(
    series: pd.Series,
    lags: Tuple[int, ...] = (1, 2, 7, 14, 30),
    rolling_windows: Tuple[int, ...] = (3, 7),
) -> pd.DataFrame:
    """
    Tabular feature matrix for daily series.
    Requires datetime index (daily).
    Output columns include:
      y, calendar features, lag_k, roll_mean_k
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")

    df = pd.DataFrame({"y": series.astype(float)})

    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for win in rolling_windows:
        df[f"roll_mean_{win}"] = df["y"].rolling(window=win).mean()

    df = df.dropna()
    return df


def process_tabular_transition(data_cfg: DataConfig):
    """
    Builds tabular features + MinMax scaling + AE loaders.

    Returns:
      loaders: (train_loader, val_loader, test_loader) yielding (x, x) where x is (B, F)
      scaler: fitted MinMaxScaler (fit on train only)
      input_shape: (F,)
      meta: dict with feature_names, indices, split points, X_all_scaled, features_df
    """
    df_raw = pd.read_csv(data_cfg.csv_path)

    # Build a proper daily series with DatetimeIndex if possible
    if data_cfg.datetime_col is not None and data_cfg.datetime_col in df_raw.columns:
        dt = pd.to_datetime(df_raw[data_cfg.datetime_col], errors="coerce")
        m = dt.notna()
        df_raw = df_raw.loc[m].copy()
        dt = dt.loc[m]
        s = pd.Series(
            df_raw[data_cfg.target_col].astype(float).to_numpy(), index=dt, name="y"
        ).sort_index()

        # if not already daily, still works; create_features_from_series uses index fields
        series = s
    else:
        # Fallback (no datetime): create dummy daily index so feature function can run
        y = df_raw[data_cfg.target_col].astype(float).to_numpy()
        idx = pd.date_range("2000-01-01", periods=len(y), freq="D")
        series = pd.Series(y, index=idx, name="y")

    # Feature engineering (tabular)
    features_df = create_features_from_series(
        series, lags=data_cfg.lags, rolling_windows=data_cfg.rolling_windows
    )
    feature_names = list(features_df.columns)
    idx_all = features_df.index

    X_all = features_df.values.astype(np.float32)  # (N, F)
    n = X_all.shape[0]

    # Time split (train/val/test) on engineered rows
    n_train = int(n * float(data_cfg.split_ratio[0]))
    n_val_end = n_train + int(n * float(data_cfg.split_ratio[1]))

    X_train = X_all[:n_train]
    X_val = X_all[n_train:n_val_end]
    X_test = X_all[n_val_end:]

    # Scale (fit on train only)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    X_all_s = scaler.transform(X_all).astype(np.float32)

    # DataLoaders for AE: (x, x)
    xtr = torch.tensor(X_train_s, dtype=torch.float32)
    xva = torch.tensor(X_val_s, dtype=torch.float32)
    xte = torch.tensor(X_test_s, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(xtr, xtr), batch_size=data_cfg.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(xva, xva), batch_size=data_cfg.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(xte, xte), batch_size=data_cfg.batch_size, shuffle=False
    )

    input_shape = int(xtr.shape[1])  # (F,)

    meta = {
        "feature_names": feature_names,
        "idx_all": idx_all,
        "n_total": n,
        "n_train": n_train,
        "n_val_end": n_val_end,
        "X_all_scaled": X_all_s,  # useful for latent forecaster fit/rollout
        "X_all": X_all,  # useful for plotting alignment
    }

    return (train_loader, val_loader, test_loader), scaler, input_shape, meta


def process_tabular_readout(data_cfg: DataConfig):
    """
    Feature engineering -> cyclic calendar encoding + onehot + scaling (via split_scale_encode_df)
    -> AE DataLoaders that yield (x, x) where x excludes y_col.

    Returns:
      loaders: (train_loader, val_loader, test_loader) yielding (x, x) with x shape (B, F_in)
      input_shape: int (F_in)
      meta: dict with:
        - n_train
        - n_val_end
        - idx_all (engineered index)
        - train_end_ts (timestamp of last train engineered row)
        - y_all_unscaled (engineered y, unscaled)
        - y_train/y_val/y_test (scaled y from split dfs)
        - scalers
        - feature_names_in
    """
    df_raw = pd.read_csv(data_cfg.csv_path)

    dt = pd.to_datetime(df_raw[data_cfg.datetime_col], errors="coerce")
    m = dt.notna()
    df_raw = df_raw.loc[m].copy()
    dt = dt.loc[m]

    series = pd.Series(
        df_raw[data_cfg.target_col].astype(float).to_numpy(),
        index=dt,
        name="y",
    ).sort_index()

    features_df = create_features_from_series(
        series, lags=data_cfg.lags, rolling_windows=data_cfg.rolling_windows
    )

    y_col = getattr(data_cfg, "y_col", "y")
    idx_all = features_df.index
    y_all_unscaled = features_df[y_col].to_numpy(dtype=np.float32)

    calendar_cols = getattr(data_cfg, "calendar_cols", [])
    nominal_cols = getattr(data_cfg, "nominal_cols", [])

    numeric_cols = getattr(data_cfg, "numeric_cols", None)
    if numeric_cols is None:
        exclude = set([y_col]) | set(calendar_cols) | set(nominal_cols)
        numeric_cols = [c for c in features_df.columns if c not in exclude]

    train_df, val_df, test_df, scalers = split_scale_encode_df(
        features_df,
        calendar_cols=calendar_cols,
        nominal_cols=nominal_cols,
        numeric_cols=numeric_cols + ([y_col] if y_col not in numeric_cols else []),
        y_col=y_col,
        split_ratio=tuple(map(float, data_cfg.split_ratio)),
        numeric_scaler=MinMaxScaler(),
        y_scaler=MinMaxScaler(),
        onehot=OneHotEncoder(handle_unknown="ignore"),
        drop_original_calendar=True,
    )

    feature_names_in = [c for c in train_df.columns if c != y_col]

    x_train = train_df[feature_names_in].to_numpy(dtype=np.float32)
    x_val = val_df[feature_names_in].to_numpy(dtype=np.float32)
    x_test = test_df[feature_names_in].to_numpy(dtype=np.float32)

    xtr = torch.tensor(x_train, dtype=torch.float32)
    xva = torch.tensor(x_val, dtype=torch.float32)
    xte = torch.tensor(x_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(xtr, xtr), batch_size=data_cfg.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(xva, xva), batch_size=data_cfg.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(xte, xte), batch_size=data_cfg.batch_size, shuffle=False
    )

    input_shape = int(xtr.shape[1])

    n_train = len(train_df)
    n_val_end = n_train + len(val_df)
    train_end_ts = idx_all[n_train - 1]

    meta = {
        "idx_all": idx_all,
        "n_train": n_train,
        "n_val_end": n_val_end,
        "train_end_ts": train_end_ts,
        "y_all_unscaled": y_all_unscaled,
        "y_train": train_df[y_col],
        "y_val": val_df[y_col],
        "y_test": test_df[y_col],
        "scalers": scalers,
        "feature_names_in": feature_names_in,
    }

    return (train_loader, val_loader, test_loader), input_shape, meta
