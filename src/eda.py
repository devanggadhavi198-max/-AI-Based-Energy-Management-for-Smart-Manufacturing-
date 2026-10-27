"""Phase 2: descriptive statistics (Table 11), Figures 7 and 8, load-type shares and the CO2 check."""
import pandas as pd

from src import config as C
from src import plots
from src.data_loader import load_clean


def descriptive_stats(df):
    """Table 11: distribution of the target plus share of near-zero intervals."""
    y = df[C.TARGET]
    d = y.describe(percentiles=[0.25, 0.5, 0.75]).rename({"50%": "median"})
    d["skewness"] = y.skew()
    d["share_below_5kWh_pct"] = 100 * (y < 5).mean()
    d["share_zero_pct"] = 100 * (y == 0).mean()
    return d.rename("Usage_kWh").to_frame()


def co2_check(df):
    """Correlation and ratio between CO2 and usage: shows why CO2 cannot be a feature."""
    return pd.DataFrame({"pearson_r": [df[C.TARGET].corr(df[C.CO2_COL])],
                         "implied_kg_per_kWh_total": [1000 * df[C.CO2_COL].sum() / df[C.TARGET].sum()],
                         "co2_distinct_values": [df[C.CO2_COL].nunique()],
                         "co2_resolution_t": [0.01]})


def load_type_table(df):
    """Share of intervals per load type, overall and during weekday daytime."""
    s = df["Load_Type"].value_counts(normalize=True).mul(100).rename("share_pct")
    hours = df.assign(h=df.index.hour).groupby("Load_Type")["h"].agg(lambda x: x.mode().iloc[0]).rename("most_common_hour")
    return pd.concat([s, hours], axis=1)


def correlations(df):
    """Pearson correlation matrix of the numeric variables (numbers behind Figure 8)."""
    return df.select_dtypes("number").corr()


def profile_summary(df):
    """Mean consumption by day type and period, and the share of energy used on weekdays 08:00-20:00."""
    start = df.index - pd.Timedelta(C.FREQ)
    day = (start.hour >= C.PEAK_HOURS[0]) & (start.hour < C.PEAK_HOURS[1])
    wk = start.dayofweek < 5
    y = df[C.TARGET]
    rows = [("mean_weekday_kwh", y[wk].mean()), ("mean_weekend_kwh", y[~wk].mean()),
            ("mean_weekday_08_20_kwh", y[wk & day].mean()), ("mean_weekday_20_08_kwh", y[wk & ~day].mean()),
            ("mean_weekend_08_20_kwh", y[~wk & day].mean()),
            ("share_energy_weekday_08_20_pct", 100 * y[wk & day].sum() / y.sum()),
            ("share_light_load_in_weekday_08_20_pct", 100 * (df.loc[wk & day, "Load_Type"] == C.FLEXIBLE_LOAD_TYPE).mean()),
            ("share_light_load_outside_weekday_08_20_pct", 100 * (df.loc[~(wk & day), "Load_Type"] == C.FLEXIBLE_LOAD_TYPE).mean()),
            ("mean_kwh_light_load", y[df["Load_Type"] == "Light_Load"].mean()),
            ("mean_kwh_medium_load", y[df["Load_Type"] == "Medium_Load"].mean()),
            ("mean_kwh_maximum_load", y[df["Load_Type"] == "Maximum_Load"].mean())]
    return pd.DataFrame(rows, columns=["item", "value"])


def main():
    df = load_clean()
    correlations(df).to_csv(C.METRICS / "eda_correlations.csv")
    profile_summary(df).to_csv(C.METRICS / "eda_profile_summary.csv", index=False)
    descriptive_stats(df).to_csv(C.METRICS / "table11_descriptive_stats.csv")
    co2 = co2_check(df); co2.to_csv(C.METRICS / "eda_co2_check.csv", index=False)
    load_type_table(df).to_csv(C.METRICS / "eda_load_type_shares.csv")
    plots.fig07_patterns(df)
    plots.fig08_correlation(df)
    print(descriptive_stats(df).round(3).to_string()); print(co2.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
