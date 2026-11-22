import random
import numpy as np
import pandas as pd
from pathlib import Path

# Fixed seed so results are reproducible (agree on this value with the team)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Find the data folder next to this file's parent folder (works wherever you run it from)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# What files are in the data folder?
print(sorted(p.name for p in DATA_DIR.iterdir()))

# Load the KO target data
ko = pd.read_csv(DATA_DIR / "target_KO.csv", index_col=0, parse_dates=True)

print("\nShape:", ko.shape)
print("First date:", ko.index.min())
print("Last date:", ko.index.max())
print("Dates in increasing order:", ko.index.is_monotonic_increasing)
print(ko.head())
# %% Step 2: Load predictors and align with KO
pred = pd.read_csv(DATA_DIR / "predictors.csv", index_col=0, parse_dates=True)

print("Predictor shape:", pred.shape)
print("Columns:", pred.columns.tolist())
print("First date:", pred.index.min())
print("Last date:", pred.index.max())
print("Dates in increasing order:", pred.index.is_monotonic_increasing)
print("Duplicate dates:", pred.index.duplicated().sum())

print("\nMissing values per column:")
print(pred.isnull().sum())

# How well do the predictor dates line up with KO's trading days?
common = ko.index.intersection(pred.index)
print("\nKO dates:", len(ko), "| predictor dates:", len(pred), "| in both:", len(common))
print("KO dates missing from predictors:", len(ko.index.difference(pred.index)))
print("Predictor dates not in KO:", len(pred.index.difference(ko.index)))
print(pred.head())
# %% Step 2b: Where are the gaps?
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

extra_dates = pred.index.difference(ko.index)
print("Predictor dates that are not KO trading days:")
print(extra_dates.tolist())

pred_ko = pred.reindex(ko.index)   # keep only KO trading days
print("\nMissing values on KO trading days:")
print(pred_ko.isnull().sum())

print("\nRows with any missing value on KO days:")
print(pred_ko[pred_ko.isnull().any(axis=1)])
# %% Step 2c: Clean predictors (KO trading days only, fill gaps from the past)
X_raw = pred.reindex(ko.index)   # drop the 7 non-KO dates
X_raw = X_raw.ffill()            # carry the last known value forward (past only)

print("Shape:", X_raw.shape)
print("Missing values left:", int(X_raw.isnull().sum().sum()))
print("Same dates as KO:", X_raw.index.equals(ko.index))

# Check the three filled days
print(X_raw.loc[["2016-10-10", "2016-11-11", "2018-01-29"]])
# %% Step 3: Target = next-day log return
log_close = np.log(ko["Close"])
y = (log_close.shift(-1) - log_close).rename("target_next_ret")

print("Target observations:", y.notna().sum())
print("Missing at the end (no 'tomorrow' yet):")
print(y[y.isnull()])

# Check one row by hand: return from 2015-01-06 close to 2015-01-07 close
manual = np.log(ko.loc["2015-01-07", "Close"] / ko.loc["2015-01-06", "Close"])
print("\ny on 2015-01-06:", y.loc["2015-01-06"])
print("Manual calculation:", manual)

print("\nTarget summary:")
print(y.describe())
# %% Step 4: Features (each uses only data up to and including day t)
ret = np.log(ko["Close"]).diff()          # r_t = log(C_t / C_{t-1})
logvol = np.log(ko["Volume"])

feat = pd.DataFrame(index=ko.index)

# Group 1: KO's own recent returns
for k in range(5):
    feat[f"ret_lag{k}"] = ret.shift(k)    # lag0 = today's return
feat["ret_mean5"] = ret.rolling(5).mean()
feat["ret_mean21"] = ret.rolling(21).mean()

# Group 2: KO volatility
feat["vol5"] = ret.rolling(5).std()
feat["vol21"] = ret.rolling(21).std()

# Group 3: KO trading volume (today vs its 21-day average, in logs)
feat["volume_ratio21"] = logvol - logvol.rolling(21).mean()

# Group 4: US market and technology
feat["SP500_ret"] = np.log(X_raw["SP500"]).diff()
feat["TECH_ret"] = np.log(X_raw["TECH"]).diff()

# Group 5: macro and commodities
feat["VIX_chg"] = X_raw["VIX"].diff()
feat["US10Y_chg"] = X_raw["US10Y"].diff()
feat["USD_ret"] = np.log(X_raw["USD"]).diff()
feat["BRENT_ret"] = np.log(X_raw["BRENT"]).diff()
feat["GOLD_ret"] = np.log(X_raw["GOLD"]).diff()

FEATURE_GROUPS = {
    "own_returns": ["ret_lag0", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4",
                    "ret_mean5", "ret_mean21"],
    "volatility": ["vol5", "vol21"],
    "volume": ["volume_ratio21"],
    "market": ["SP500_ret", "TECH_ret"],
    "macro": ["VIX_chg", "US10Y_chg", "USD_ret", "BRENT_ret", "GOLD_ret"],
}

# Remove the warm-up rows where the rolling windows are not full yet
X = feat.dropna()

print("Features:", X.shape[1], "| rows:", X.shape[0])
print("First date:", X.index[0], "| last date:", X.index[-1])
print("Infinite values:", int(np.isinf(X.values).sum()))
print("Missing values:", int(X.isnull().sum().sum()))

# Timing check: today's return in the features must equal yesterday's target
check = np.isclose(X.loc["2015-02-10", "ret_lag0"], y.loc["2015-02-09"])
print("Timing check (ret_lag0 today == target yesterday):", check)

print("\nSummary of each feature:")
print(X.describe().T[["mean", "std", "min", "max"]])