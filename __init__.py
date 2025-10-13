# Make ts_benchmark importable as a top-level module for compatibility
import sys
import importlib

try:
    _ts_benchmark = importlib.import_module("paper_icps.TFB.ts_benchmark")
    sys.modules["ts_benchmark"] = _ts_benchmark
except Exception as e:
    print("Warning: Could not register ts_benchmark alias:", e)