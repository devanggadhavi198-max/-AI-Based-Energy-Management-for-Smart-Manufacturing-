"""Persistence and seasonal-naive reference forecasts on the test partition."""
from src import config as C
from src.evaluate import metrics, save_metric_row, save_predictions
from src.splits import load_splits


def persistence(X):
    """y_hat(t+1) = y(t)."""
    return X["y_lag0"].to_numpy()


def seasonal_naive(X):
    """y_hat(t+1) = y(t-95): same interval on the previous day."""
    return X[f"y_lag{C.SEASONAL_LAG}"].to_numpy()


def main():
    _, _, te = load_splits()
    for name, f in (("Persistence", persistence), ("SeasonalNaive", seasonal_naive)):
        yhat = f(te)
        m = metrics(te["target"], yhat)
        save_predictions(name, te["target_time"], te["target"], yhat)
        save_metric_row(name, m, 0.0, {"family": "Baseline"})
        print(name, {k: round(v, 4) for k, v in m.items()})


if __name__ == "__main__":
    main()
