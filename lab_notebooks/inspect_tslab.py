import sys, inspect, pkgutil, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("Files in tslab:", sorted(p.name for p in (ROOT / "tslab").glob("*.py")))
print("requirements.txt exists:", (ROOT / "requirements.txt").exists())

from tslab import data, features
for fn in (data.load_market_data, features.make_dataset):
    print("\n=====", fn.__module__ + "." + fn.__name__)
    print(inspect.getsource(fn))

# Find any walk-forward / splitting helpers in the package
import tslab
for m in pkgutil.walk_packages(tslab.__path__, "tslab."):
    try:
        mod = importlib.import_module(m.name)
    except Exception as e:
        print("\nCould not import", m.name, "->", repr(e))
        continue
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if fn.__module__ == m.name and any(k in name.lower() for k in ("walk", "split", "fold")):
            print("\n=====", m.name + "." + name)
            print(inspect.getsource(fn))