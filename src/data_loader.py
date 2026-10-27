"""Load the raw UCI steel dataset, repair timestamps, validate it, and write the clean copy plus Table 4."""
import hashlib

import pandas as pd

from src import config as C


def sha256(path):
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_raw(path=C.RAW_CSV):
    """Read the raw CSV exactly as published (raw file is never modified)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_timestamps(df):
    """Parse day-first dates; the 00:00 reading closes each day, so it belongs to the next calendar day."""
    ts = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0)
    ts = ts.where(~midnight, ts + pd.Timedelta(days=1))
    out = df.drop(columns="date").copy()
    out.insert(0, "timestamp", ts)
    return out.sort_values("timestamp").set_index("timestamp")


def validate(df):
    """Run the structural checks; return a dict of results and raise if any fails."""
    idx = df.index
    steps = idx.to_series().diff().dropna()
    checks = {
        "rows": len(df),
        "rows_expected": C.EXPECTED_ROWS,
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "irregular_steps": int((steps != pd.Timedelta(C.FREQ)).sum()),
        "missing_values": int(df.isna().sum().sum()),
        "negative_usage": int((df[C.TARGET] < 0).sum()),
        "midnight_relabelled": int(((idx.hour == 0) & (idx.minute == 0)).sum()),
        "start": idx.min(),
        "end": idx.max(),
    }
    assert checks["rows"] == C.EXPECTED_ROWS, checks
    assert checks["duplicate_timestamps"] == 0, checks
    assert checks["irregular_steps"] == 0, checks
    assert checks["missing_values"] == 0, checks
    assert checks["negative_usage"] == 0, checks
    return checks


def overview_table(df, checks):
    """Table 4: dataset overview with per-column type, range and missing count."""
    rows = [
        ("Source", f"UCI Machine Learning Repository, ID {C.UCI_ID}", ""),
        ("Licence", "CC BY 4.0", ""),
        ("Records", checks["rows"], ""),
        ("Period", f"{checks['start']} to {checks['end']}", ""),
        ("Resolution", C.FREQ, ""),
        ("Duplicate timestamps", checks["duplicate_timestamps"], ""),
        ("Irregular time steps", checks["irregular_steps"], ""),
        ("Missing values", checks["missing_values"], ""),
        ("Midnight readings moved to next day", checks["midnight_relabelled"], "00:00 reading closes the previous day"),
    ]
    head = pd.DataFrame(rows, columns=["item", "value", "note"])
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            cols.append((c, "numeric", s.min(), s.max(), int(s.isna().sum())))
        else:
            cols.append((c, "categorical", "|".join(sorted(s.unique())), "", int(s.isna().sum())))
    cols = pd.DataFrame(cols, columns=["variable", "type", "min_or_levels", "max", "missing"])
    return head, cols


def load_clean():
    """Return the cleaned, validated dataframe (reads the interim copy if present)."""
    if C.CLEAN_CSV.exists():
        return pd.read_csv(C.CLEAN_CSV, parse_dates=["timestamp"], index_col="timestamp")
    return parse_timestamps(read_raw())


def main():
    digest = sha256(C.RAW_CSV)
    assert digest == C.RAW_SHA256, f"raw file changed: {digest}"
    df = parse_timestamps(read_raw())
    checks = validate(df)
    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    C.METRICS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_CSV)
    head, cols = overview_table(df, checks)
    head.to_csv(C.METRICS / "table04_dataset_overview.csv", index=False)
    cols.to_csv(C.METRICS / "table04b_variables.csv", index=False)
    print(head.to_string(index=False))
    print(cols.to_string(index=False))


if __name__ == "__main__":
    main()
