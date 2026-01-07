import matplotlib.pyplot as plt


def plot_series(series, train_ratio=0.8):
    n = len(series)
    n_train = int(train_ratio * n)

    # Razbij serijo na train + test
    train_series = series.iloc[:n_train]
    test_series = series.iloc[n_train:]

    plt.figure(figsize=(14, 6))

    # Train: modra, tanka
    plt.plot(
        train_series.index,
        train_series.values,
        color="blue",
        linewidth=1,
        label="Train (true)",
    )

    # Test: zelena, tanka
    plt.plot(
        test_series.index,
        test_series.values,
        color="green",
        linewidth=1,
        label="Test (true)",
    )

    plt.title("Time series")
    plt.xlabel("Datum")
    plt.ylabel("Poraba")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast(series, results, train_ratio=0.8):
    """
    Nariše:
    - modro: dejanske vrednosti v učnem delu
    - zeleno: dejanske vrednosti v testnem delu
    - rdeče: napovedi
    """
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    effective_horizon = results["effective_horizon"].iloc[0]

    n = len(series)
    n_train = int(train_ratio * n)

    # Razbij serijo na train + test
    train_series = series.iloc[:n_train]
    test_series = series.iloc[n_train : n_train + effective_horizon]

    plt.figure(figsize=(14, 6))

    # Train: modra, tanka
    plt.plot(
        train_series.index,
        train_series.values,
        color="blue",
        linewidth=1,
        label="Train (true)",
    )

    # Test: zelena, tanka
    plt.plot(
        test_series.index,
        test_series.values,
        color="green",
        linewidth=1,
        label="Test (true)",
    )

    # Napovedi: rdeča, debelejša
    plt.plot(test_series.index, y_pred, color="red", linewidth=2.5, label="Napoved")

    # plt.title(f"Forecast results (latent_dim={results['latent_dim']}, horizon={results['forecast_horizon']})")
    plt.title("Forecast results")
    plt.xlabel("Datum")
    plt.ylabel("Poraba")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast_comparison(
    series, latent_results, baseline_results, train_ratio: float = 0.8
):
    """
    Nariše skupen graf:
      - modro: dejanske vrednosti (train)
      - zeleno: dejanske vrednosti (test)
      - rdeče: latentna napoved
      - oranžno, črtkano: baseline napoved
    """

    # Skupni eval horizont = min obeh
    h_latent = latent_results["effective_horizon"]
    h_base = baseline_results["effective_horizon"]
    horizon = min(h_latent, h_base)

    # Resnične vrednosti (vzamemo iz latent ali baseline, morajo biti iste v prvih 'horizon' korakih)
    y_true = latent_results["y_true"][:horizon]
    y_pred_latent = latent_results["y_pred"][:horizon]
    y_pred_base = baseline_results["y_pred"][:horizon]

    # Train/test razbitje indeksa
    n = len(series)
    n_train = int(train_ratio * n)

    train_series = series.iloc[:n_train]
    # test del, ki ga dejansko uporabljamo za primerjavo
    test_series = series.iloc[n_train : n_train + horizon]

    plt.figure(figsize=(14, 6))

    # 1) Train del: modra tanka črta
    plt.plot(
        train_series.index,
        train_series.values,
        color="blue",
        linewidth=1,
        label="Train (true)",
    )

    # 2) Test del: zelena tanka črta
    plt.plot(
        test_series.index,
        test_series.values,
        color="green",
        linewidth=1,
        label="Test (true)",
    )

    # 3) Latentna napoved: rdeča debelejša črta
    plt.plot(
        test_series.index,
        y_pred_latent,
        color="red",
        linewidth=2.5,
        label="Napoved (latentni prostor)",
    )

    # 4) Baseline napoved: oranžna debelejša črtkana črta
    plt.plot(
        test_series.index,
        y_pred_base,
        color="orange",
        linewidth=2.5,
        label="Napoved (baseline)",
    )

    plt.title(
        "Primerjava napovedi: latentni model vs baseline\n"
        f"latent_dim={latent_results.get('latent_dim', 'N/A')}, "
        f"horizon_latent={latent_results['forecast_horizon']}, "
        f"horizon_baseline={baseline_results['forecast_horizon']}, "
        f"prikazan horizon={horizon}"
    )
    plt.xlabel("Datum")
    plt.ylabel("Poraba")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
