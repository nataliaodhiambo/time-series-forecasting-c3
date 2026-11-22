import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.linear_model import Ridge
from tslab import validation
from ko_setup import make_ko_setup, RIDGE_ALPHA

s = make_ko_setup()
X, y, splits, dataset = s["X"], s["y"], s["splits"], s["dataset"]

# Out-of-sample predictions of the next-day log return (scaler fitted per training block)
pred = validation.walk_forward_predict(lambda: Ridge(alpha=RIDGE_ALPHA), X, y, splits)
mask = pred.notna()
actual, p = y[mask], pred[mask]
print("Test days scored:", int(mask.sum()),
      "| from", actual.index[0].date(), "to", actual.index[-1].date())

def scores(actual, forecast):
    err = actual - forecast
    return {
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE": float(err.abs().mean()),
        # R2 against the naive benchmark (naive predicts a return of 0)
        "OOS_R2": float(1 - (err ** 2).sum() / (actual ** 2).sum()),
        "Directional_acc": float((np.sign(forecast) == np.sign(actual)).mean()),
    }

naive = pd.Series(0.0, index=actual.index)
table = pd.DataFrame({"Ridge": scores(actual, p), "Naive": scores(actual, naive)}).T
pd.set_option("display.width", 200)
print(table.round(6))

# Convert to prices: forecast close = today's close * exp(predicted return)
close_t = dataset.loc[actual.index, "close"]
target_close = dataset.loc[actual.index, "target_close"]
price_ridge = close_t * np.exp(p)
price_naive = close_t
print("\nPrice error (USD):")
print("Ridge MAE:", round(float((target_close - price_ridge).abs().mean()), 4),
      "| Naive MAE:", round(float((target_close - price_naive).abs().mean()), 4))

print("\nPrediction spread (a tiny spread means the model stays close to naive):")
print(p.describe().round(6))