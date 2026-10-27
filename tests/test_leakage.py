"""No feature may use information from after time t; CO2 and contemporaneous electrical values are never inputs."""
import numpy as np
import pandas as pd

from src import config as C
from src.features import feature_columns, sequence_windows, tabular_features
from src.models.baseline import persistence, seasonal_naive


def perturb_future(df, t0, seed=0):
    """Scramble every value strictly after t0."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    fut = out.index > t0
    for c in out.select_dtypes("number").columns:
        out.loc[fut, c] = rng.permutation(out.loc[fut, c].to_numpy())
    out.loc[fut, "Load_Type"] = rng.permutation(out.loc[fut, "Load_Type"].to_numpy())
    return out


def test_features_ignore_the_future(clean):
    t0 = clean.index[20_000]
    a, b = tabular_features(clean), tabular_features(perturb_future(clean, t0))
    cols = feature_columns(a)
    past = a.index[a.index <= t0]
    pd.testing.assert_frame_equal(a.loc[past, cols], b.loc[past, cols])
    # the target of row t0 is y(t0+1): it must change, proving the target really is in the future
    assert a.loc[t0, "target"] != b.loc[t0, "target"] or a.loc[t0, "target"] == 0


def test_target_is_next_interval(clean):
    X = tabular_features(clean)
    assert ((X["target_time"] - X.index) == pd.Timedelta(C.FREQ)).all()
    assert np.allclose(X["target"], clean.loc[X["target_time"], C.TARGET].to_numpy())
    assert np.allclose(X["y_lag0"], clean.loc[X.index, C.TARGET].to_numpy())


def test_no_forbidden_columns(clean):
    cols = feature_columns(tabular_features(clean))
    assert C.CO2_COL not in cols and not any("CO2" in c for c in cols)
    assert not set(cols) & set(C.ELECTRICAL_COLS + [C.TARGET, "Load_Type", "WeekStatus", "Day_of_week"])


def test_lags_stay_inside_the_window():
    assert max(C.TARGET_LAGS) < C.WINDOW and max(C.ROLLING_WINDOWS) <= C.WINDOW


def test_sequence_windows_end_at_t(clean):
    X = tabular_features(clean)
    times = X.index[[0, 5000, len(X) - 1]]
    w = sequence_windows(clean, times)
    assert w.shape == (3, C.WINDOW, w.shape[2])
    assert np.allclose(w[:, -1, 0], clean.loc[times, C.TARGET].to_numpy())           # last step is y(t)
    assert np.allclose(w[:, 0, 0], clean[C.TARGET].shift(C.WINDOW - 1).loc[times].to_numpy())
    t0 = X.index[20_000]
    w2 = sequence_windows(perturb_future(clean, t0), X.index[X.index <= t0][-50:])
    assert np.allclose(sequence_windows(clean, X.index[X.index <= t0][-50:]), w2)


def test_baselines_use_only_the_past(clean):
    X = tabular_features(clean)
    assert np.allclose(persistence(X), clean[C.TARGET].loc[X.index].to_numpy())
    assert np.allclose(seasonal_naive(X), clean[C.TARGET].shift(C.SEASONAL_LAG).loc[X.index].to_numpy())
