"""Linear Regression, Random Forest and XGBoost with a shared interface; small grid on the validation set only."""
import itertools
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src import config as C
from src.evaluate import Timer, metrics, save_metric_row, save_predictions
from src.features import feature_columns
from src.splits import fit_scaler, load_splits


class TabularModel:
    """Common fit/predict/get_params wrapper; scaling is fitted on training data inside fit()."""

    def __init__(self, name, factory, grid=None, scale=False):
        self.name, self.factory, self.grid, self.scale = name, factory, grid or {}, scale
        self.model, self.scaler, self.best_params = None, None, {}

    def _prep(self, X, fit=False):
        if not self.scale:
            return X
        if fit:
            self.scaler = fit_scaler(X)
        return self.scaler.transform(X)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        """Grid search on (Xva, yva) by validation MAE, then keep the best model fitted on training data."""
        Xtr_ = self._prep(Xtr, fit=True)
        keys = list(self.grid)
        best = (np.inf, None, None)
        for values in itertools.product(*self.grid.values()) if keys else [()]:
            params = dict(zip(keys, values))
            m = self.factory(**params).fit(Xtr_, ytr)
            score = metrics(yva, m.predict(self._prep(Xva)))["MAE"] if Xva is not None else 0.0
            if score < best[0]:
                best = (score, params, m)
        self.val_mae, self.best_params, self.model = best
        return self

    def predict(self, X):
        return self.model.predict(self._prep(X))

    def get_params(self):
        return self.best_params


def build_models():
    """The three tabular models with their search grids from config."""
    return [
        TabularModel("LinearRegression", LinearRegression, scale=True),
        TabularModel("RandomForest", lambda **p: RandomForestRegressor(random_state=C.SEED, n_jobs=C.N_JOBS, **p), C.RF_GRID),
        TabularModel("XGBoost", lambda **p: XGBRegressor(random_state=C.SEED, n_jobs=C.N_JOBS, **C.XGB_FIXED, **p), C.XGB_GRID),
    ]


def main():
    tr, va, te = load_splits()
    cols = feature_columns(tr)
    C.MODELS.mkdir(exist_ok=True)
    params_out = {}
    for mdl in build_models():
        with Timer() as t_search:
            mdl.fit(tr[cols], tr["target"], va[cols], va["target"])
        # training time of the selected configuration alone (refit), reported in Table 12
        with Timer() as t_fit:
            final = TabularModel(mdl.name, mdl.factory, {k: [v] for k, v in mdl.best_params.items()}, mdl.scale)
            final.fit(tr[cols], tr["target"], va[cols], va["target"])
        yhat = final.predict(te[cols])
        m = metrics(te["target"], yhat)
        save_predictions(mdl.name, te["target_time"], te["target"], yhat)
        save_metric_row(mdl.name, m, t_fit.seconds, {"family": "ML", "val_MAE": final.val_mae, "search_seconds": t_search.seconds})
        joblib.dump({"model": final.model, "scaler": final.scaler, "features": cols}, C.MODELS / f"{mdl.name}.joblib")
        params_out[mdl.name] = {"selected": final.get_params(), "grid": {k: list(v) for k, v in mdl.grid.items()}}
        print(mdl.name, final.get_params(), {k: round(v, 4) for k, v in m.items()}, f"fit {t_fit.seconds:.1f}s")
    (C.METRICS / "ml_hyperparameters.json").write_text(json.dumps(params_out, indent=1, default=str))


if __name__ == "__main__":
    main()
