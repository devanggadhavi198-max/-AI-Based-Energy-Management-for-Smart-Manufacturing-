"""Metrics for every model and the master comparison table (Table 12) with Figures 9 and 10."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config as C


def metrics(y, yhat):
    """MAE, RMSE, R2 and CV(RMSE) in percent. MAPE is excluded on purpose."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    return {"MAE": float(mean_absolute_error(y, yhat)), "RMSE": rmse,
            "R2": float(r2_score(y, yhat)), "CV_RMSE_pct": 100 * rmse / float(np.mean(y))}


def save_predictions(name, times, y, yhat, seed=None):
    """Write one model's test predictions to results/predictions/."""
    C.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_seed{seed}" if seed is not None else name
    pd.DataFrame({"target_time": times, "y_true": y, "y_pred": yhat}).to_csv(C.PREDICTIONS / f"{tag}.csv", index=False)


def save_metric_row(name, m, train_seconds, extra=None):
    """Append/replace one model's metric row in results/metrics/model_rows/."""
    d = C.METRICS / "model_rows"
    d.mkdir(parents=True, exist_ok=True)
    row = {"model": name, **m, "train_seconds": train_seconds, **(extra or {})}
    (d / f"{name}.json").write_text(json.dumps(row, indent=1, default=str))


class Timer:
    """Wall-clock timer used for training-time reporting."""
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0


ORDER = ["Persistence", "SeasonalNaive", "LinearRegression", "RandomForest", "XGBoost", "LSTM", "BiLSTM"]


def performance_table():
    """Table 12: one row per model, baseline first, with improvement over persistence."""
    rows = [json.loads((C.METRICS / "model_rows" / f"{m}.json").read_text()) for m in ORDER
            if (C.METRICS / "model_rows" / f"{m}.json").exists()]
    t = pd.DataFrame(rows)
    ref = t.set_index("model").loc["Persistence"]
    t["MAE_improvement_vs_persistence_pct"] = 100 * (ref["MAE"] - t["MAE"]) / ref["MAE"]
    t["RMSE_improvement_vs_persistence_pct"] = 100 * (ref["RMSE"] - t["RMSE"]) / ref["RMSE"]
    keep = ["model", "family", "MAE", "MAE_std", "RMSE", "RMSE_std", "R2", "R2_std", "CV_RMSE_pct", "CV_RMSE_pct_std",
            "train_seconds", "train_seconds_std", "MAE_improvement_vs_persistence_pct", "RMSE_improvement_vs_persistence_pct"]
    return t[[c for c in keep if c in t.columns]]


def hyperparameter_table():
    """Table 10: selected hyperparameters for all five models."""
    ml = json.loads((C.METRICS / "ml_hyperparameters.json").read_text())
    rows = [(m, k, v, ", ".join(map(str, d["grid"].get(k, []))) or "-") for m, d in ml.items() for k, v in d["selected"].items()]
    if not ml.get("LinearRegression", {}).get("selected"):
        rows.append(("LinearRegression", "fit_intercept", True, "-"))
    for m in ("LSTM", "BiLSTM"):
        rows += [(m, "recurrent_layers", 1, "-"), (m, "units", C.DL_UNITS, "-"), (m, "dropout", C.DL_DROPOUT, "-"),
                 (m, "optimizer", "Adam", "-"), (m, "learning_rate", C.DL_LEARNING_RATE, "-"), (m, "batch_size", C.DL_BATCH_SIZE, "-"),
                 (m, "loss", "MSE", "-"), (m, "max_epochs", C.DL_MAX_EPOCHS, "-"), (m, "early_stopping_patience", C.DL_PATIENCE, "-"),
                 (m, "window_steps", C.WINDOW, "-"), (m, "seeds", " ".join(map(str, C.DL_SEEDS)), "-")]
    return pd.DataFrame(rows, columns=["model", "hyperparameter", "value", "searched_values"])


def _sysctl(key):
    """macOS hardware field via sysctl; empty string elsewhere."""
    import subprocess
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def run_info():
    """Seeds, library versions and hardware, so every number is traceable."""
    import platform, sklearn, xgboost, matplotlib
    from importlib.metadata import version
    return pd.DataFrame([("seed", C.SEED), ("dl_seeds", " ".join(map(str, C.DL_SEEDS))), ("python", platform.python_version()),
                         ("pandas", pd.__version__), ("numpy", np.__version__), ("scikit-learn", sklearn.__version__),
                         ("xgboost", xgboost.__version__), ("tensorflow", version("tensorflow")), ("matplotlib", matplotlib.__version__),
                         ("keras", version("keras")), ("seaborn", version("seaborn")), ("pytest", version("pytest")), ("streamlit", version("streamlit")), ("plotly", version("plotly")),
                         ("platform", platform.platform()), ("machine", platform.machine()), ("cpu", _sysctl("machdep.cpu.brand_string")),
                         ("cpu_cores", _sysctl("hw.ncpu")), ("ram_gb", round(int(_sysctl("hw.memsize") or 0) / 2**30)), ("gpu", "none (CPU only)")],
                        columns=["item", "value"])


def feature_importance():
    """Impurity importance (RF) and gain importance (XGBoost) from the saved models; no retraining."""
    import joblib
    out = []
    for name in ("RandomForest", "XGBoost"):
        b = joblib.load(C.MODELS / f"{name}.joblib")
        imp = pd.Series(b["model"].feature_importances_, index=b["features"])
        out.append((imp / imp.sum()).rename(name))
    return pd.concat(out, axis=1).sort_values("RandomForest", ascending=False)


def error_analysis(preds):
    """MAE by hour of day and by regime: ramp-up, ramp-down, stable, and the top 5 % of actual load."""
    hour, regime = {}, []
    for name, p in preds.items():
        p = p.copy()
        p["ae"] = (p["y_true"] - p["y_pred"]).abs()
        p["prev"] = p["y_true"].shift(1)
        hour[name] = p.groupby(p["target_time"].dt.hour)["ae"].mean()
        step = p["y_true"] - p["prev"]
        high = p["y_true"] >= p["y_true"].quantile(0.95)
        for lab, mask in (("ramp_up_>20kWh", step > 20), ("ramp_down_>20kWh", step < -20),
                          ("stable_|step|<=20kWh", step.abs() <= 20), ("top_5pct_load", high), ("all", p["ae"].notna())):
            regime.append({"model": name, "regime": lab, "intervals": int(mask.sum()), "MAE": float(p.loc[mask, "ae"].mean())})
    return pd.DataFrame(hour).rename_axis("hour"), pd.DataFrame(regime)


def main():
    from src import plots
    t = performance_table()
    t.to_csv(C.METRICS / "table12_model_performance.csv", index=False)
    hyperparameter_table().to_csv(C.METRICS / "table10_hyperparameters.csv", index=False)
    run_info().to_csv(C.METRICS / "run_info.csv", index=False)
    best = t[t["family"] != "Baseline"].sort_values("MAE").iloc[0]["model"]
    preds = {m: pd.read_csv(C.PREDICTIONS / f"{m}.csv", parse_dates=["target_time"]) for m in t["model"] if m != "SeasonalNaive"}
    plots.fig09_actual_vs_predicted(preds[best], best)
    plots.fig10_error_by_hour(preds)
    feature_importance().to_csv(C.METRICS / "feature_importance.csv")
    by_hour, by_regime = error_analysis(preds)
    by_hour.to_csv(C.METRICS / "error_by_hour.csv")
    by_regime.to_csv(C.METRICS / "error_by_regime.csv", index=False)
    over = t[(t["family"] != "Baseline") & (t["R2"] > 0.98)]
    if len(over):
        print("WARNING: R2 above 0.98 -> check for leakage:", list(over["model"]))
    print(t.round(3).to_string(index=False)); print("best:", best)


if __name__ == "__main__":
    main()
