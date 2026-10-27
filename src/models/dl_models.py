"""LSTM and BiLSTM on 96-step windows of the same variables; three seeds each, early stopping on validation."""
import json
import os
import random

import tensorflow as tf  # imported before scikit-learn/xgboost: avoids an OpenMP clash on macOS
import numpy as np

from src import config as C
from src.data_loader import load_clean
from src.evaluate import Timer, metrics, save_metric_row, save_predictions
from src.features import SEQ_COLS, sequence_windows
from src.splits import load_splits


def set_seeds(seed):
    """Seed numpy, random and tensorflow for one run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build(kind, n_features):
    """One recurrent layer (plain or bidirectional), dropout, dense output; Adam and MSE."""
    L = tf.keras.layers
    rnn = L.LSTM(C.DL_UNITS)
    inp = L.Input(shape=(C.WINDOW, n_features))
    h = L.Bidirectional(rnn)(inp) if kind == "BiLSTM" else rnn(inp)
    h = L.Dropout(C.DL_DROPOUT)(h)
    out = L.Dense(1)(h)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(C.DL_LEARNING_RATE), loss="mse")
    return model


def scale_sequences(tr, va, te):
    """Standardise each sequence feature with mean/std from the training windows' last step only."""
    last = tr[:, -1, :]
    mu, sd = last.mean(0), last.std(0)
    sd[sd == 0] = 1.0
    return [((a - mu) / sd).astype("float32") for a in (tr, va, te)], (mu, sd)


def main():
    df = load_clean()
    tr, va, te = load_splits()
    Xs = [sequence_windows(df, p.index) for p in (tr, va, te)]
    (Xtr, Xva, Xte), (mu, sd) = scale_sequences(*Xs)
    ymu, ysd = tr["target"].mean(), tr["target"].std()
    ytr, yva = ((tr["target"] - ymu) / ysd).to_numpy("float32"), ((va["target"] - ymu) / ysd).to_numpy("float32")
    C.MODELS.mkdir(exist_ok=True)
    info = {"window": C.WINDOW, "features": SEQ_COLS, "train_shape": list(Xtr.shape), "dtype": str(Xtr.dtype)}
    for kind in ("LSTM", "BiLSTM"):
        runs = []
        for seed in C.DL_SEEDS:
            set_seeds(seed)
            model = build(kind, Xtr.shape[2])
            stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.DL_PATIENCE, restore_best_weights=True)
            with Timer() as t:
                hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=C.DL_MAX_EPOCHS,
                                 batch_size=C.DL_BATCH_SIZE, callbacks=[stop], shuffle=True, verbose=0)
            yhat = model.predict(Xte, verbose=0).ravel() * ysd + ymu
            m = metrics(te["target"], yhat)
            save_predictions(kind, te["target_time"], te["target"], yhat, seed=seed)
            model.save(C.MODELS / f"{kind}_seed{seed}.keras")
            runs.append({**m, "train_seconds": t.seconds, "epochs": len(hist.history["loss"]), "seed": seed, "yhat": yhat})
            print(kind, seed, {k: round(v, 4) for k, v in m.items()}, f"{t.seconds:.1f}s, {len(hist.history['loss'])} epochs")
        mean_pred = np.mean([r["yhat"] for r in runs], axis=0)
        save_predictions(kind, te["target_time"], te["target"], mean_pred)   # seed-average, used by the scenario/app
        keys = ["MAE", "RMSE", "R2", "CV_RMSE_pct", "train_seconds"]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        agg.update({f"{k}_std": float(np.std([r[k] for r in runs], ddof=1)) for k in keys})
        save_metric_row(kind, {k: agg[k] for k in keys[:4]}, agg["train_seconds"],
                        {"family": "DL", **{k: v for k, v in agg.items() if k.endswith("_std")},
                         "seeds": C.DL_SEEDS, "epochs": [r["epochs"] for r in runs]})
    info["scaler_mean"], info["scaler_std"], info["y_mean"], info["y_std"] = mu.tolist(), sd.tolist(), float(ymu), float(ysd)
    (C.MODELS / "dl_preprocessing.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
