# %% Role 2 pipeline: dataset and walk-forward folds
import sys, random
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tslab import config, data, features, validation

random.seed(config.SEED)
np.random.seed(config.SEED)

# %% Step 6: dataset
frame = data.load_market_data()
dataset = features.make_dataset(frame, horizon=1)

print("Frame rows:", len(frame))
print("Dataset rows:", len(dataset),
      "| first:", dataset.index[0].date(), "| last:", dataset.index[-1].date())
print("Number of features:", len(features.ALL_FEATURES))
print("Rows with no target yet:", int(dataset["target_return"].isna().sum()))

# Training/testing table: drop the last row (no 'tomorrow' yet)
model_data = dataset.dropna(subset=["target_return"])
X = model_data[features.ALL_FEATURES]
y = model_data["target_return"]
print("Model table:", X.shape, "| y rows:", len(y))
print("Missing values in X:", int(X.isnull().sum().sum()))

# %% Walk-forward folds (expanding window, 1-day horizon so gap = 0)
splits = list(validation.walk_forward_splits(
    n_samples=len(X),
    initial_train=config.INITIAL_TRAIN,
    test_size=config.TEST_SIZE,
    gap=0,
))
print("\nNumber of folds:", len(splits))

table = validation.describe_splits(splits, X.index)
pd.set_option("display.width", 200)
print(table.head(3))
print(table.tail(3))

# %% Checks: time order is respected
for train_idx, test_idx in splits:
    assert train_idx.max() < test_idx.min(), "Training rows overlap the test rows"
all_test = np.concatenate([t for _, t in splits])
assert len(all_test) == len(set(all_test)), "A row is tested more than once"
assert (np.diff(all_test) == 1).all(), "Test rows are not consecutive"
print("\nAll checks passed. Test rows:", len(all_test),
      "| first test date:", X.index[all_test[0]].date())
# %% Step 7: Look-ahead test
# (a) The team's feature set must pass
result = features.check_no_lookahead(frame, n_checks=20, seed=config.SEED)
print("build_features passes the look-ahead test:", result)

# (b) Control: a feature that deliberately peeks at tomorrow must FAIL
def cheating_features(df):
    out = features.build_features(df)
    out["cheat_tomorrow_ret"] = np.log(df["Close"].shift(-1) / df["Close"])
    return out

try:
    features.check_no_lookahead(frame, feature_function=cheating_features,
                                n_checks=20, seed=config.SEED)
    print("PROBLEM: the cheating feature was NOT caught")
except AssertionError as e:
    print("Cheating feature correctly caught:")
    print(" ", e)
    # %% Step 8: Tune Ridge alpha using ONLY rows before the first test day
    from sklearn.linear_model import Ridge

    first_test_row = splits[0][1][0]  # 1000
    X_tune, y_tune = X.iloc[:first_test_row], y.iloc[:first_test_row]
    assert X_tune.index[-1] < X.index[first_test_row]
    print("Tuning data:", X_tune.index[0].date(), "to", X_tune.index[-1].date(),
          "| first real test day:", X.index[first_test_row].date())

    # Smaller walk-forward inside the tuning block
    tune_splits = list(validation.walk_forward_splits(
        len(X_tune), initial_train=500, test_size=config.TEST_SIZE, gap=0))
    print("Tuning folds:", len(tune_splits))

    rows = []
    for alpha in [0.1, 1, 10, 100, 1000, 10000]:
        pred = validation.walk_forward_predict(
            lambda a=alpha: Ridge(alpha=a), X_tune, y_tune, tune_splits)
        mask = pred.notna()
        rmse = float(np.sqrt(((y_tune[mask] - pred[mask]) ** 2).mean()))
        rows.append({"alpha": alpha, "rmse": rmse})

    naive_rmse = float(np.sqrt((y_tune[mask] ** 2).mean()))  # naive = predict a 0 return
    results = pd.DataFrame(rows).set_index("alpha")
    results["vs_naive_%"] = (results["rmse"] / naive_rmse - 1) * 100
    print("\nTuning rows scored:", int(mask.sum()))
    print("Naive RMSE:", round(naive_rmse, 6))
    print(results.round(6))

    best_alpha = results["rmse"].idxmin()
    print("\nBest alpha on the tuning data:", best_alpha)