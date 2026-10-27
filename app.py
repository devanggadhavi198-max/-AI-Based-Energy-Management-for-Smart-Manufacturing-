"""Streamlit prototype (Phase 10). Loads saved artefacts only; the scenario sliders call src.scenario.run_scenario."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.scenario import run_scenario, scenario_frame

st.set_page_config(page_title="Predictive Energy Management", layout="wide")


@st.cache_data
def load_table(name):
    return pd.read_csv(C.METRICS / f"{name}.csv")


@st.cache_data
def load_clean():
    return pd.read_csv(C.CLEAN_CSV, parse_dates=["timestamp"], index_col="timestamp")


@st.cache_data
def load_pred(model):
    return pd.read_csv(C.PREDICTIONS / f"{model}.csv", parse_dates=["target_time"])


@st.cache_data
def training_ceiling():
    s = pd.read_csv(C.SPLITS_CSV, parse_dates=["last_target_time"])
    clean = load_clean()
    return float(clean.loc[: s.loc[s.partition == "train", "last_target_time"].iloc[0], C.TARGET].max())


perf = load_table("table12_model_performance")
models = [m for m in perf["model"] if m != "SeasonalNaive"]
best = perf[perf["family"] != "Baseline"].sort_values("MAE").iloc[0]["model"]

PAGES = ["Data Overview", "Model Comparison", "Forecast Viewer", "Energy Management Scenario"]
start = {"data": 0, "models": 1, "forecast": 2, "scenario": 3}.get(st.query_params.get("page", ""), 0)
page = st.sidebar.radio("Page", PAGES, index=start)
st.sidebar.caption("Demonstrator on historical data (UCI Steel Industry Energy Consumption). No model is trained here.")

if page == "Data Overview":
    st.title("Data overview")
    c1, c2 = st.columns([1, 2])
    c1.subheader("Descriptive statistics")
    desc = load_table("table11_descriptive_stats").round(3)
    c1.table(desc.rename(columns={desc.columns[0]: "Statistic"}).set_index("Statistic"))
    c2.subheader("Average daily profile")
    c2.image(str(C.FIGURES / "fig07_consumption_patterns.png"))
    st.subheader("Correlation heatmap")
    st.image(str(C.FIGURES / "fig08_correlation_heatmap.png"), width=760)

elif page == "Model Comparison":
    st.title("Model comparison (test partition)")
    cols = ["model", "family", "MAE", "MAE_std", "RMSE", "RMSE_std", "R2", "CV_RMSE_pct", "train_seconds", "MAE_improvement_vs_persistence_pct"]
    st.dataframe(perf[cols].round(3).map(lambda v: "–" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v))), hide_index=True, use_container_width=True)
    long = perf.melt(id_vars="model", value_vars=["MAE", "RMSE"], var_name="metric", value_name="kWh")
    st.plotly_chart(px.bar(long, x="model", y="kWh", color="metric", barmode="group"), use_container_width=True)
    st.caption(f"Best model by test MAE: **{best}**. LSTM/BiLSTM values are means over seeds {C.DL_SEEDS}.")

elif page == "Forecast Viewer":
    st.title("Forecast viewer")
    model = st.selectbox("Model", models, index=models.index(best))
    p = load_pred(model)
    lo, hi = p["target_time"].min().date(), p["target_time"].max().date()
    start, end = st.date_input("Date range", (pd.Timestamp(C.SAMPLE_WEEK_START).date(), (pd.Timestamp(C.SAMPLE_WEEK_START) + pd.Timedelta(days=7)).date()), min_value=lo, max_value=hi)
    q = p[(p["target_time"].dt.date >= start) & (p["target_time"].dt.date <= end)]
    fig = go.Figure([go.Scatter(x=q.target_time, y=q.y_true, name="Actual", line=dict(color="grey")),
                     go.Scatter(x=q.target_time, y=q.y_pred, name=f"Predicted ({model})", line=dict(color="#1f77b4"))])
    fig.update_layout(yaxis_title="kWh per 15 min", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.scatter(q, x="target_time", y=q.y_true - q.y_pred, labels={"y": "Residual [kWh]"}, height=260), use_container_width=True)

else:
    st.title("Energy management scenario")
    st.caption(f"Forecasts from **{best}**. Every change re-runs `src.scenario.run_scenario`, the same function that produced the thesis scenario results.")
    c = st.columns(4)
    share = c[0].slider("Shiftable share of flexible load", 0.0, 0.40, C.SHIFTABLE_SHARE, 0.05, format="%.2f")
    peak = c[1].slider("Peak tariff [EUR/kWh]", 0.10, 0.60, C.TARIFF_PEAK_EUR_KWH, 0.01)
    off = c[2].slider("Off-peak tariff [EUR/kWh]", 0.05, 0.40, C.TARIFF_OFFPEAK_EUR_KWH, 0.01)
    ef = c[3].slider("Emission factor [kg CO2/kWh]", 0.10, 0.90, C.EMISSION_FACTOR_KG_KWH, 0.01)
    frame = scenario_frame(C.PREDICTIONS / f"{best}.csv", load_clean())
    f, res = run_scenario(frame, share=share, peak_price=peak, offpeak_price=off, emission_factor=ef, ceiling=training_ceiling())
    b, s = res.iloc[0], res.iloc[1]
    m = st.columns(4)
    m[0].metric("Energy [kWh]", f"{s.energy_kwh:,.0f}", f"{s.energy_kwh - b.energy_kwh:+.0f}")
    m[1].metric("Cost [EUR]", f"{s.cost_eur:,.0f}", f"{100 * (s.cost_eur - b.cost_eur) / b.cost_eur:+.2f} %", delta_color="inverse")
    m[2].metric("CO2 [t]", f"{s.co2_t:,.2f}", f"{s.co2_t - b.co2_t:+.2f}", delta_color="inverse")
    m[3].metric("Shifted energy [kWh]", f"{s.shifted_energy_kwh:,.0f}")
    days = sorted(f["day"].dt.date.unique())
    d = st.select_slider("Day", days, value=pd.Timestamp(C.SAMPLE_DAY).date())
    g = f[f["day"].dt.date == d]
    fig = go.Figure([go.Scatter(x=g.index, y=g.baseline, name="Baseline (actual)", line=dict(color="grey", shape="hv")),
                     go.Scatter(x=g.index, y=g.shifted, name="Shifted", line=dict(color="#1f77b4", shape="hv"))])
    fig.update_layout(yaxis_title="kWh per 15 min", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Constant emission factor: shifting conserves energy, so CO2 changes only if the factor varies over time.")
