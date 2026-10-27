"""Energy management scenario: baseline vs rule-based load shifting driven by the predicted profile.

`run_scenario` is the single implementation used by the pipeline and by the Streamlit app.
"""
import numpy as np
import pandas as pd

from src import config as C


def peak_mask(index, peak_hours=C.PEAK_HOURS):
    """True for weekday intervals whose start hour lies in [start, end)."""
    start = index - pd.Timedelta(C.FREQ)          # a reading at 08:15 covers 08:00-08:15
    return (start.dayofweek < 5) & (start.hour >= peak_hours[0]) & (start.hour < peak_hours[1])


def fill_valleys(level_base, headroom, total):
    """Water-filling: raise the lowest intervals of level_base to a common level so the added energy equals total."""
    lo, hi = level_base.min(), level_base.max() + headroom.max() + total
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.minimum(np.clip(mid - level_base, 0, None), headroom).sum() < total:
            lo = mid
        else:
            hi = mid
    add = np.minimum(np.clip(hi - level_base, 0, None), headroom)
    return add * (total / add.sum()) if add.sum() > 0 else add


def shift_day(actual, predicted, is_peak, flexible, share, ceiling):
    """Move share*min(pred, actual) of flexible peak load into the valleys of the day's predicted off-peak profile."""
    remove = np.where(is_peak & flexible, share * np.minimum(np.clip(predicted, 0, None), actual), 0.0)
    headroom = np.where(~is_peak, np.clip(ceiling - actual, 0, None), 0.0)
    total, room = remove.sum(), headroom.sum()
    if total <= 0 or room <= 0:
        return actual.copy()
    if room < total:                               # cannot place it all: shift less, never create energy
        remove *= room / total
        total = remove.sum()
    base = np.where(~is_peak, np.clip(predicted, 0, None), np.inf)
    add = fill_valleys(np.where(np.isfinite(base), base, 0.0), headroom, total)
    return actual - remove + add


def run_scenario(frame, share=C.SHIFTABLE_SHARE, peak_price=C.TARIFF_PEAK_EUR_KWH, offpeak_price=C.TARIFF_OFFPEAK_EUR_KWH,
                 emission_factor=C.EMISSION_FACTOR_KG_KWH, peak_hours=C.PEAK_HOURS, ceiling=None):
    """frame: index = interval end time, columns y_true, y_pred, load_type. Returns (interval table, summary table)."""
    assert 0 <= share <= 1
    ceiling = frame["y_true"].max() if ceiling is None else ceiling
    f = frame.copy()
    f["is_peak"] = peak_mask(f.index, peak_hours)
    f["flexible"] = f["load_type"] == C.FLEXIBLE_LOAD_TYPE
    day = (f.index - pd.Timedelta(C.FREQ)).normalize()
    shifted = np.empty(len(f))
    for d, pos in pd.Series(np.arange(len(f)), index=day).groupby(level=0):
        p = pos.to_numpy()
        shifted[p] = shift_day(f["y_true"].to_numpy()[p], f["y_pred"].to_numpy()[p], f["is_peak"].to_numpy()[p],
                               f["flexible"].to_numpy()[p], share, ceiling)
    f["baseline"], f["shifted"] = f["y_true"], shifted
    f["price"] = np.where(f["is_peak"], peak_price, offpeak_price)
    f["day"] = day
    rows = []
    for name in ("baseline", "shifted"):
        load = f[name]
        rows.append({"scenario": name.capitalize(), "energy_kwh": load.sum(), "cost_eur": (load * f["price"]).sum(),
                     "co2_t": load.sum() * emission_factor / 1000, "peak_demand_kw": load.max() * 4,
                     "peak_window_energy_kwh": load[f["is_peak"]].sum()})
    res = pd.DataFrame(rows)
    res["shifted_energy_kwh"] = [0.0, float((f["baseline"] - f["shifted"]).clip(lower=0).sum())]
    return f, res


def scenario_frame(pred_path, clean):
    """Join saved test predictions with the load type of each interval."""
    p = pd.read_csv(pred_path, parse_dates=["target_time"]).set_index("target_time")
    p["load_type"] = clean.loc[p.index, "Load_Type"]
    return p


def change_table(res):
    """Absolute and relative change shifted vs baseline."""
    b, s = res.iloc[0], res.iloc[1]
    cols = ["energy_kwh", "cost_eur", "co2_t", "peak_demand_kw", "peak_window_energy_kwh"]
    return pd.DataFrame({"metric": cols, "baseline": [b[c] for c in cols], "shifted": [s[c] for c in cols],
                         "change": [s[c] - b[c] for c in cols], "change_pct": [100 * (s[c] - b[c]) / b[c] for c in cols]})


def assumptions_table():
    """Table 7: every scenario assumption, straight from config."""
    return pd.DataFrame([
        ("Tariff structure", f"two-band time-of-use: peak {C.TARIFF_PEAK_EUR_KWH} EUR/kWh, off-peak {C.TARIFF_OFFPEAK_EUR_KWH} EUR/kWh"),
        ("Peak window", f"weekdays {C.PEAK_HOURS[0]:02d}:00-{C.PEAK_HOURS[1]:02d}:00"),
        ("Emission factor", f"{C.EMISSION_FACTOR_KG_KWH} kg CO2/kWh, constant"),
        ("Flexible load", f"intervals with Load_Type = {C.FLEXIBLE_LOAD_TYPE} inside the peak window"),
        ("Shiftable share (headline)", f"{C.SHIFTABLE_SHARE:.0%} of min(predicted, actual) flexible load"),
        ("Temporal constraint", "reallocation within the same calendar day only"),
        ("Capacity constraint", "no interval above the maximum observed in the training partition"),
        ("Reallocation rule", "valley filling: off-peak intervals of the same day raised to a common predicted level"),
        ("Sensitivity: shares", ", ".join(f"{s:.0%}" for s in C.SENSITIVITY_SHARES)),
        ("Sensitivity: tariffs", "; ".join(f"{a}/{b} EUR/kWh" for a, b in C.SENSITIVITY_TARIFFS)),
    ], columns=["parameter", "assumption"])


def main():
    from src import plots
    from src.data_loader import load_clean
    from src.splits import load_splits
    clean = load_clean()
    tr, _, _ = load_splits()
    ceiling = tr["target"].max()
    rows = pd.read_csv(C.METRICS / "table12_model_performance.csv")
    best = rows[rows["family"] != "Baseline"].sort_values("MAE").iloc[0]["model"]
    frame = scenario_frame(C.PREDICTIONS / f"{best}.csv", clean)
    f, res = run_scenario(frame, ceiling=ceiling)
    daily = f.groupby("day")[["baseline", "shifted"]].sum()
    assert np.allclose(daily["baseline"], daily["shifted"]), "daily energy not conserved"
    res.insert(0, "model", best)
    res.to_csv(C.METRICS / "table13_scenario_results.csv", index=False)
    change_table(res.drop(columns="model")).to_csv(C.METRICS / "table13_scenario_change.csv", index=False)
    assumptions_table().to_csv(C.METRICS / "table07_assumptions.csv", index=False)
    sens = []
    for share in C.SENSITIVITY_SHARES:
        for peak, off in C.SENSITIVITY_TARIFFS:
            _, r = run_scenario(frame, share=share, peak_price=peak, offpeak_price=off, ceiling=ceiling)
            b, s = r.iloc[0], r.iloc[1]
            sens.append({"share": share, "peak_eur_kwh": peak, "offpeak_eur_kwh": off,
                         "cost_baseline_eur": b.cost_eur, "cost_shifted_eur": s.cost_eur,
                         "cost_saving_eur": b.cost_eur - s.cost_eur, "cost_saving_pct": 100 * (b.cost_eur - s.cost_eur) / b.cost_eur,
                         "co2_change_t": s.co2_t - b.co2_t, "shifted_energy_kwh": s.shifted_energy_kwh})
    pd.DataFrame(sens).to_csv(C.METRICS / "table13b_sensitivity.csv", index=False)
    pd.DataFrame([("model", best), ("test_days", f["day"].nunique()), ("intervals", len(f)),
                  ("peak_intervals", int(f["is_peak"].sum())), ("flexible_peak_intervals", int((f["is_peak"] & f["flexible"]).sum())),
                  ("flexible_peak_energy_kwh", float(f.loc[f["is_peak"] & f["flexible"], "baseline"].sum())),
                  ("ceiling_kwh", float(ceiling)), ("days_with_shift", int((f.groupby("day").apply(lambda g: (g["baseline"] != g["shifted"]).any())).sum())),
                  ("max_daily_energy_error_kwh", float((daily["baseline"] - daily["shifted"]).abs().max()))],
                 columns=["item", "value"]).to_csv(C.METRICS / "table13c_scenario_diagnostics.csv", index=False)
    value = []
    for name in list(rows[rows["model"] != "SeasonalNaive"]["model"]) + ["PerfectForecast"]:
        fr = frame.copy()
        fr["y_pred"] = fr["y_true"] if name == "PerfectForecast" else scenario_frame(C.PREDICTIONS / f"{name}.csv", clean)["y_pred"]
        _, r = run_scenario(fr, ceiling=ceiling)
        value.append({"forecast": name, "shifted_energy_kwh": r.iloc[1].shifted_energy_kwh,
                      "cost_saving_eur": r.iloc[0].cost_eur - r.iloc[1].cost_eur,
                      "cost_saving_pct": 100 * (r.iloc[0].cost_eur - r.iloc[1].cost_eur) / r.iloc[0].cost_eur})
    pd.DataFrame(value).to_csv(C.METRICS / "table13d_forecast_value.csv", index=False)
    plots.fig11_shifted_day(f.loc[f["day"] == pd.Timestamp(C.SAMPLE_DAY)])
    plots.fig12_cost_co2(res)
    print(res.round(3).to_string(index=False)); print(pd.DataFrame(sens).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
