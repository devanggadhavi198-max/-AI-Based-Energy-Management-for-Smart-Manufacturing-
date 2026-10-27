"""Load shifting conserves daily energy, respects the share cap and the capacity ceiling."""
import numpy as np
import pandas as pd

from src import config as C
from src.scenario import peak_mask, run_scenario


def frame_from(clean, start, days, noise=0.1, seed=0):
    idx = clean.loc[start:].index[: days * C.STEPS_PER_DAY]
    y = clean.loc[idx, C.TARGET]
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"y_true": y, "y_pred": y * (1 + noise * rng.standard_normal(len(y))), "load_type": clean.loc[idx, "Load_Type"]})


def test_daily_energy_conserved(clean):
    f, res = run_scenario(frame_from(clean, "2018-11-05 00:15", 14))
    d = f.groupby("day")[["baseline", "shifted"]].sum()
    assert np.allclose(d["baseline"], d["shifted"])
    assert np.isclose(res.loc[0, "energy_kwh"], res.loc[1, "energy_kwh"])


def test_share_cap_and_ceiling(clean):
    share, ceiling = 0.3, 60.0
    f, _ = run_scenario(frame_from(clean, "2018-11-05 00:15", 14), share=share, ceiling=ceiling)
    removed = (f["baseline"] - f["shifted"]).clip(lower=0)
    assert (removed <= share * f["baseline"] + 1e-9).all()
    assert (removed[~(f["is_peak"] & f["flexible"])] == 0).all()
    added = (f["shifted"] - f["baseline"]).clip(lower=0)
    assert (f.loc[added > 0, "shifted"] <= ceiling + 1e-9).all()
    assert (added[f["is_peak"]] == 0).all()
    assert (f["shifted"] >= -1e-12).all()


def test_zero_share_changes_nothing(clean):
    f, res = run_scenario(frame_from(clean, "2018-11-05 00:15", 7), share=0.0)
    assert np.allclose(f["baseline"], f["shifted"])
    assert np.isclose(res.loc[0, "cost_eur"], res.loc[1, "cost_eur"])


def test_shifting_never_raises_cost(clean):
    _, res = run_scenario(frame_from(clean, "2018-11-05 00:15", 14))
    assert res.loc[1, "cost_eur"] <= res.loc[0, "cost_eur"] + 1e-9


def test_peak_window_weekdays_only():
    idx = pd.date_range("2018-12-08 00:15", periods=2 * 96, freq="15min")    # Saturday + Sunday
    assert not peak_mask(idx).any()
    wk = pd.date_range("2018-12-10 08:15", periods=4, freq="15min")          # Monday morning
    assert peak_mask(wk).all()
