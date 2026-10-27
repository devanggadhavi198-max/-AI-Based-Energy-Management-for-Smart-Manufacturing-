"""All paths, seeds and parameters for the thesis pipeline. No magic numbers elsewhere."""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

RAW_CSV = DATA_RAW / "Steel_industry_data.csv"
RAW_SHA256 = "9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc"
RAW_URL = "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip"
UCI_ID = 851
CLEAN_CSV = DATA_INTERIM / "steel_clean.csv"
FEATURES_CSV = DATA_PROCESSED / "features.csv"
SPLITS_CSV = METRICS / "split_boundaries.csv"

# --- Dataset ---------------------------------------------------------------
EXPECTED_ROWS = 35_040
STEPS_PER_DAY = 96
FREQ = "15min"
TARGET = "Usage_kWh"
CO2_COL = "CO2(tCO2)"                       # reporting quantity only, never a feature
ELECTRICAL_COLS = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
]
CATEGORICAL_COLS = ["WeekStatus", "Day_of_week", "Load_Type"]
LOAD_TYPES = ["Light_Load", "Medium_Load", "Maximum_Load"]
FLEXIBLE_LOAD_TYPE = "Light_Load"

# --- Forecasting setup -----------------------------------------------------
HORIZON = 1                                  # t+1, 15 minutes ahead
WINDOW = 96                                  # 24 h history for every model
TARGET_LAGS = [0, 1, 2, 3, 95]               # y_t, y_{t-1}, y_{t-2}, y_{t-3}, y_{t-95}: all inside the 96-step window
ROLLING_WINDOWS = [4, 96]
SEASONAL_LAG = 95                            # seasonal naive: y_hat(t+1) = y(t-95)
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# --- Reproducibility -------------------------------------------------------
SEED = 42
DL_SEEDS = [42, 43, 44]

# --- Machine learning grids (validation set only) --------------------------
RF_GRID = {"n_estimators": [200, 400], "max_depth": [10, 20, None], "min_samples_leaf": [1, 5]}
XGB_GRID = {"n_estimators": [300, 600], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]}
XGB_FIXED = {"subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist"}
N_JOBS = -1

# --- Deep learning ---------------------------------------------------------
DL_UNITS = 64
DL_DROPOUT = 0.2
DL_LEARNING_RATE = 1e-3
DL_BATCH_SIZE = 256
DL_MAX_EPOCHS = 60
DL_PATIENCE = 6

# --- Energy management scenario (assumptions -> table07) -------------------
TARIFF_PEAK_EUR_KWH = 0.30
TARIFF_OFFPEAK_EUR_KWH = 0.15
PEAK_HOURS = (8, 20)                         # [start, end) local hour, weekdays only
EMISSION_FACTOR_KG_KWH = 0.42                # constant; equals the factor implied by the dataset CO2 column (results/metrics/eda_co2_check.csv)
SHIFTABLE_SHARE = 0.20                       # headline cap on flexible load moved
SENSITIVITY_SHARES = [0.10, 0.20, 0.30]
SENSITIVITY_TARIFFS = [(0.25, 0.15), (0.35, 0.10)]   # (peak, off-peak): narrow and wide spread
SCENARIO_MODEL = "best"                      # use the best model by test MAE

# --- Figures ---------------------------------------------------------------
FIG_SIZE = (6.3, 3.4)                        # inches, fits a 15 cm text width
FIG_SIZE_TALL = (6.3, 4.6)
FONT_SIZE = 9
DPI_PNG = 200
SAMPLE_WEEK_START = "2018-12-10"             # representative test week (Monday)
SAMPLE_DAY = "2018-12-12"                    # representative weekday for Figure 11
