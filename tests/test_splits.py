"""Chronological 70/15/15 split with no overlap and a train-only scaler."""
import numpy as np

from src import config as C
from src.features import feature_columns, tabular_features
from src.splits import chrono_split, fit_scaler


def test_split_is_chronological_and_disjoint(clean):
    X = tabular_features(clean)
    tr, va, te = chrono_split(X)
    assert len(tr) + len(va) + len(te) == len(X)
    assert tr.index.max() < va.index.min() and va.index.max() < te.index.min()
    assert tr["target_time"].max() <= va.index.min() and va["target_time"].max() <= te.index.min()
    assert not (set(tr.index) & set(va.index) or set(va.index) & set(te.index))
    for part, r in zip((tr, va, te), C.SPLIT_RATIOS):
        assert abs(len(part) / len(X) - r) < 0.001


def test_split_refuses_unsorted_input(clean):
    X = tabular_features(clean).sample(frac=1, random_state=0)
    try:
        chrono_split(X)
    except AssertionError:
        return
    raise AssertionError("shuffled input was accepted")


def test_scaler_uses_training_rows_only(clean):
    X = tabular_features(clean)
    tr, va, _ = chrono_split(X)
    cols = feature_columns(X)
    sc = fit_scaler(tr[cols])
    assert np.allclose(sc.mean_, tr[cols].mean().to_numpy())
    assert not np.allclose(sc.mean_, X[cols].mean().to_numpy())
