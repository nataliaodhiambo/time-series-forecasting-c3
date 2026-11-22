"""Shared setup for the KO forecasting lab (built by Role 2).

Usage in another script in this folder:
    from ko_setup import make_ko_setup, RIDGE_ALPHA
    s = make_ko_setup()
    X, y, splits = s["X"], s["y"], s["splits"]
"""
import sys, random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tslab import config, data, features, validation

# Tuned on rows before 2019-10-08 only (role2_pipeline.py, Step 8), for horizon=1.
# Roles that forecast other horizons should re-tune on pre-test data.
RIDGE_ALPHA = 1000


def make_ko_setup(horizon=1):
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    frame = data.load_market_data(verbose=False)
    dataset = features.make_dataset(frame, horizon=horizon)

    # Rows with no target yet (the last `horizon` rows) are kept in `dataset`
    # for the real forecast, but removed from the training/testing table.
    model_data = dataset.dropna(subset=["target_return"])
    X = model_data[features.ALL_FEATURES]
    y = model_data["target_return"]

    splits = list(validation.walk_forward_splits(
        n_samples=len(X),
        initial_train=config.INITIAL_TRAIN,
        test_size=config.TEST_SIZE,
        gap=horizon - 1,
    ))
    return {"frame": frame, "dataset": dataset, "X": X, "y": y, "splits": splits}


if __name__ == "__main__":
    s = make_ko_setup()
    X, y, splits = s["X"], s["y"], s["splits"]
    print("X:", X.shape, "| y:", len(y), "| folds:", len(splits))
    print("First test date:", X.index[splits[0][1][0]].date())
    print("RIDGE_ALPHA:", RIDGE_ALPHA)