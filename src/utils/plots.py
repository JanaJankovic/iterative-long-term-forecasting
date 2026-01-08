import matplotlib.pyplot as plt
import numpy as np


def plot_forecast_simple(
    series: np.ndarray, y_pred: np.ndarray, train_ratio: float = 0.8
):
    series = np.asarray(series).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    n = series.shape[0]
    n_train = int(train_ratio * n)
    H = y_pred.shape[0]

    y_test = series[n_train : n_train + H]

    if y_test.shape[0] != H:
        raise ValueError(
            f"Not enough points after split: need {H}, got {y_test.shape[0]}"
        )

    x_test = np.arange(n_train, n_train + H)

    plt.figure(figsize=(14, 6))

    plt.plot(np.arange(n_train), series[:n_train], label="Train (true)", linewidth=1)
    plt.plot(x_test, y_test, label="Test (true)", linewidth=1)
    plt.plot(x_test, y_pred, label="Forecast", linewidth=2.5)

    plt.title("Forecast results")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
