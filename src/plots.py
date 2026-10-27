"""Every thesis figure, saved as vector PDF (LaTeX) and PNG (preview) with one consistent style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as C

plt.rcParams.update({
    "font.size": C.FONT_SIZE, "axes.titlesize": C.FONT_SIZE, "axes.labelsize": C.FONT_SIZE,
    "xtick.labelsize": C.FONT_SIZE - 1, "ytick.labelsize": C.FONT_SIZE - 1, "legend.fontsize": C.FONT_SIZE - 1,
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#7f7f7f"


def save(fig, name):
    """Write results/figures/<name>.pdf and .png."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"{name}.pdf")
    fig.savefig(C.FIGURES / f"{name}.png", dpi=C.DPI_PNG)
    plt.close(fig)


def fig07_patterns(df):
    """Figure 7: average daily profile, weekday vs weekend."""
    d = df.assign(hour=df.index.hour + df.index.minute / 60, wk=np.where(df["WeekStatus"] == "Weekend", "Weekend", "Weekday"))
    prof = d.groupby(["wk", "hour"])[C.TARGET].mean().unstack(0)
    fig, ax = plt.subplots(figsize=C.FIG_SIZE)
    ax.plot(prof.index, prof["Weekday"], color=BLUE, label="Weekday")
    ax.plot(prof.index, prof["Weekend"], color=ORANGE, label="Weekend")
    ax.plot(prof.index, d.groupby("hour")[C.TARGET].mean(), color=GREY, ls="--", lw=1, label="All days")
    ax.set(xlabel="Hour of day", ylabel="Mean consumption per 15 min [kWh]", xlim=(0, 24), xticks=range(0, 25, 3))
    ax.legend(frameon=False)
    save(fig, "fig07_consumption_patterns")


def fig08_correlation(df):
    """Figure 8: correlation heatmap of the numeric variables."""
    num = df.select_dtypes("number").rename(columns=lambda c: c.replace("_Current", "").replace("_Reactive", "_React").replace(".Power", "_Power"))
    fig, ax = plt.subplots(figsize=C.FIG_SIZE_TALL)
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, annot_kws={"size": C.FONT_SIZE - 2}, cbar_kws={"shrink": 0.8})
    ax.tick_params(axis="x", rotation=40)
    for t in ax.get_xticklabels(): t.set_ha("right")
    save(fig, "fig08_correlation_heatmap")


def fig09_actual_vs_predicted(pred, name):
    """Figure 9: actual vs predicted over the representative test week."""
    p = pred.set_index("target_time").loc[C.SAMPLE_WEEK_START:pd.Timestamp(C.SAMPLE_WEEK_START) + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=C.FIG_SIZE)
    ax.plot(p.index, p["y_true"], color=GREY, lw=1.1, label="Actual")
    ax.plot(p.index, p["y_pred"], color=BLUE, lw=0.9, label=f"Predicted ({name})")
    ax.set(ylabel="Consumption per 15 min [kWh]")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d.%m."))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig09_actual_vs_predicted")


def fig10_error_by_hour(preds):
    """Figure 10: mean absolute error by hour of day for each model (what the table cannot show)."""
    fig, ax = plt.subplots(figsize=C.FIG_SIZE)
    for name, p in preds.items():
        e = (p["y_true"] - p["y_pred"]).abs().groupby(pd.to_datetime(p["target_time"]).dt.hour).mean()
        ax.plot(e.index, e.values, marker="o", ms=2.5, lw=1, label=name)
    ax.set(xlabel="Hour of day (target interval)", ylabel="Mean absolute error [kWh]", xticks=range(0, 24, 3))
    ax.legend(frameon=False, ncol=2)
    save(fig, "fig10_error_by_hour")


def fig11_shifted_day(day):
    """Figure 11: baseline vs shifted load profile for one weekday, peak window shaded."""
    fig, ax = plt.subplots(figsize=C.FIG_SIZE)
    t = day.index
    ax.axvspan(t[0].normalize() + pd.Timedelta(hours=C.PEAK_HOURS[0]), t[0].normalize() + pd.Timedelta(hours=C.PEAK_HOURS[1]),
               color=ORANGE, alpha=0.08, label="Peak tariff window")
    ax.step(t, day["baseline"], where="post", color=GREY, lw=1.1, label="Baseline (actual)")
    ax.step(t, day["shifted"], where="post", color=BLUE, lw=1.0, label="Shifted")
    ax.set(ylabel="Consumption per 15 min [kWh]")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig11_baseline_vs_shifted")


def fig12_cost_co2(res):
    """Figure 12: total cost and CO2 of both scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=C.FIG_SIZE)
    for ax, col, lab in ((axes[0], "cost_eur", "Energy cost [EUR]"), (axes[1], "co2_t", "CO$_2$ emissions [t]")):
        v = res.set_index("scenario")[col]
        bars = ax.bar(v.index, v.values, color=[GREY, BLUE], width=0.55)
        ax.bar_label(bars, fmt="%.0f" if col == "cost_eur" else "%.1f", fontsize=C.FONT_SIZE - 1)
        ax.set(ylabel=lab, ylim=(0, v.max() * 1.15))
    save(fig, "fig12_cost_co2")
