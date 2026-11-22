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