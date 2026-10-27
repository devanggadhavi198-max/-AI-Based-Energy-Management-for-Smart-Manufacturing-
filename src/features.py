"""Build the tabular feature matrix and the sequence inputs; every feature uses information up to t only."""
import numpy as np
import pandas as pd

from src import config as C
from src.data_loader import load_clean

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def step_features(df):
    """Per-interval inputs observable at time t (used as the last step of the sequence and by tree models)."""
    out = pd.DataFrame(index=df.index)
    out["y"] = df[C.TARGET]
    for c in C.ELECTRICAL_COLS:
        out[c] = df[c]
    tod = df["NSM"] / 86_400.0
    out["tod_sin"] = np.sin(2 * np.pi * tod)
    out["tod_cos"] = np.cos(2 * np.pi * tod)
    out["hour"] = df.index.hour
    out["nsm"] = df["NSM"]
    out["month"] = df.index.month
    out["is_weekend"] = (df["WeekStatus"] == "Weekend").astype(int)
    for d in DOW:
        out[f"dow_{d}"] = (df["Day_of_week"] == d).astype(int)
    for lt in C.LOAD_TYPES:
        out[f"load_{lt}"] = (df["Load_Type"] == lt).astype(int)
    return out


def tabular_features(df):
    """Tree/LR features at row t plus the target y(t+1); drops warm-up and last rows."""
    s = step_features(df)
    X = s.drop(columns="y").add_suffix("_t")
    for k in C.TARGET_LAGS:
        X[f"y_lag{k}"] = s["y"].shift(k)
    for w in C.ROLLING_WINDOWS:
        X[f"y_roll_mean{w}"] = s["y"].rolling(w).mean()
        X[f"y_roll_std{w}"] = s["y"].rolling(w).std()
    X["target"] = s["y"].shift(-C.HORIZON)
    X["target_time"] = pd.Series(df.index, index=df.index).shift(-C.HORIZON)
    return X.dropna()


def feature_columns(X):
    """Names of model inputs (everything except target bookkeeping)."""
    return [c for c in X.columns if c not in ("target", "target_time")]


SEQ_COLS = ["y"] + C.ELECTRICAL_COLS + ["tod_sin", "tod_cos", "is_weekend", "month"] + \
           [f"dow_{d}" for d in DOW] + [f"load_{lt}" for lt in C.LOAD_TYPES]


def sequence_windows(df, row_times, window=C.WINDOW):
    """3D array (samples, window, features): steps t-window+1 ... t for each prediction time t."""
    s = step_features(df)[SEQ_COLS].astype("float32")
    pos = s.index.get_indexer(row_times)
    assert (pos >= window - 1).all(), "window would start before the data"
    arr = s.to_numpy()
    idx = pos[:, None] + np.arange(-window + 1, 1)[None, :]
    return arr[idx]


FEATURE_NOTES = {
    "y_lag0": ("Autoregressive", "y(t), last observed value"),
    "y_lag1": ("Autoregressive", "y(t-1)"), "y_lag2": ("Autoregressive", "y(t-2)"), "y_lag3": ("Autoregressive", "y(t-3)"),
    "y_lag95": ("Autoregressive", "y(t-95): same time yesterday relative to t+1"),
    "y_roll_mean4": ("Rolling", "mean of y(t-3..t)"), "y_roll_std4": ("Rolling", "std of y(t-3..t)"),
    "y_roll_mean96": ("Rolling", "mean of y(t-95..t)"), "y_roll_std96": ("Rolling", "std of y(t-95..t)"),
}


def feature_table(cols):
    """Table 5: every feature with its group and justification."""
    rows = []
    for c in cols:
        if c in FEATURE_NOTES:
            g, j = FEATURE_NOTES[c]
        elif any(c.startswith(e) for e in C.ELECTRICAL_COLS):
            g, j = "Lagged electrical", "measured at t, i.e. one step before the target; the t+1 value is never used"
        elif c.startswith("dow_") or c in ("hour_t", "nsm_t", "month_t", "is_weekend_t", "tod_sin_t", "tod_cos_t"):
            g, j = "Calendar", "deterministic from the timestamp at t"
        elif c.startswith("load_"):
            g, j = "Operational (lagged)", "load type at t, lagged so no t+1 information enters"
        else:
            g, j = "Other", ""
        rows.append((c, g, j))
    rows.append((C.CO2_COL, "Excluded", "derived from Usage_kWh; would leak the target"))
    return pd.DataFrame(rows, columns=["feature", "group", "justification"])


def main():
    df = load_clean()
    X = tabular_features(df)
    cols = feature_columns(X)
    assert C.CO2_COL not in X.columns and not any("CO2" in c for c in cols)
    assert not any(c in cols for c in C.ELECTRICAL_COLS), "contemporaneous electrical column in features"
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X.to_csv(C.FEATURES_CSV)
    feature_table(cols).to_csv(C.METRICS / "table05_features.csv", index=False)
    print(f"feature matrix {X.shape}, {len(cols)} features, rows from {X.index.min()} to {X.index.max()}")


if __name__ == "__main__":
    main()
