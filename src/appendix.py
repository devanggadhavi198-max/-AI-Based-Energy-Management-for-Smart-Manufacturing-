"""Tables for Appendix B: per-seed deep learning results, validation errors and top feature importances."""
import json

import pandas as pd

from src import config as C
from src.evaluate import metrics


def per_seed():
    """Test metrics of every LSTM/BiLSTM run, from the saved per-seed predictions."""
    rows = []
    for kind in ("LSTM", "BiLSTM"):
        epochs = json.loads((C.METRICS / "model_rows" / f"{kind}.json").read_text())["epochs"]
        for seed, ep in zip(C.DL_SEEDS, epochs):
            p = pd.read_csv(C.PREDICTIONS / f"{kind}_seed{seed}.csv")
            rows.append({"model": kind, "seed": seed, "epochs": ep, **metrics(p["y_true"], p["y_pred"])})
        p = pd.read_csv(C.PREDICTIONS / f"{kind}.csv")
        rows.append({"model": kind, "seed": "average of predictions", "epochs": "", **metrics(p["y_true"], p["y_pred"])})
    return pd.DataFrame(rows)


def validation_errors():
    """Validation MAE of the selected tabular configurations and grid-search time."""
    rows = []
    for m in ("LinearRegression", "RandomForest", "XGBoost"):
        r = json.loads((C.METRICS / "model_rows" / f"{m}.json").read_text())
        rows.append({"model": m, "val_MAE": r["val_MAE"], "search_seconds": r["search_seconds"], "fit_seconds": r["train_seconds"]})
    return pd.DataFrame(rows)


def main():
    per_seed().to_csv(C.METRICS / "appendixB_dl_per_seed.csv", index=False)
    validation_errors().to_csv(C.METRICS / "appendixB_validation.csv", index=False)
    fi = pd.read_csv(C.METRICS / "feature_importance.csv", index_col=0).head(10)
    fi.to_csv(C.METRICS / "appendixB_feature_importance_top10.csv")
    print(per_seed().round(3).to_string(index=False)); print(validation_errors().round(3).to_string(index=False)); print(fi.round(3))


if __name__ == "__main__":
    main()
