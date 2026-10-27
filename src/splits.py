"""Chronological 70/15/15 split of the feature rows and a scaler fitted on training data only."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config as C


def split_index(n, ratios=C.SPLIT_RATIOS):
    """Return (train_end, val_end) positions for a chronological split of n rows."""
    assert abs(sum(ratios) - 1) < 1e-9
    train_end = int(round(n * ratios[0]))
    val_end = int(round(n * (ratios[0] + ratios[1])))
    return train_end, val_end


def chrono_split(X):
    """Split a time-indexed frame into train/val/test without shuffling."""
    assert X.index.is_monotonic_increasing, "rows must be in time order"
    a, b = split_index(len(X))
    return X.iloc[:a], X.iloc[a:b], X.iloc[b:]


def fit_scaler(train_X):
    """StandardScaler fitted on the training partition only."""
    return StandardScaler().fit(train_X)


def boundaries(train, val, test):
    """Table of first/last prediction time and size for each partition."""
    rows = [(n, len(p), p.index.min(), p.index.max(), p["target_time"].min(), p["target_time"].max())
            for n, p in (("train", train), ("validation", val), ("test", test))]
    return pd.DataFrame(rows, columns=["partition", "rows", "first_t", "last_t", "first_target_time", "last_target_time"])


def load_splits():
    """Read the feature matrix and return the three partitions."""
    X = pd.read_csv(C.FEATURES_CSV, parse_dates=["timestamp", "target_time"], index_col="timestamp")
    return chrono_split(X)


def main():
    tr, va, te = load_splits()
    b = boundaries(tr, va, te)
    b.to_csv(C.SPLITS_CSV, index=False)
    print(b.to_string(index=False))


if __name__ == "__main__":
    main()
