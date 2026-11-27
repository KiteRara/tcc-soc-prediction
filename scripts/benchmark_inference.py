"""Benchmark reprodutível de latência de inferência dos modelos RF e XGBoost em CPU."""
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.common import DATA_PROCESSED, FEATURES_COMPAT, FEATURES_FULL, MODELS, save_json


def bench(model, X, n_single=200, n_batch=10000):
    model.n_jobs = 1
    x1 = X[:1]
    for _ in range(20):
        model.predict(x1)
    t0 = time.perf_counter()
    for _ in range(n_single):
        model.predict(x1)
    ms_single = (time.perf_counter() - t0) / n_single * 1000.0

    Xb = X[:n_batch]
    t0 = time.perf_counter()
    model.predict(Xb)
    ms_batch_per_sample = (time.perf_counter() - t0) / len(Xb) * 1000.0
    return {"single_sample_ms": round(ms_single, 4), "batch_per_sample_ms": round(ms_batch_per_sample, 6)}


def run():
    te = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    X_f = joblib.load(MODELS / "scaler_full.joblib").transform(te[FEATURES_FULL])
    X_c = joblib.load(MODELS / "scaler_compat.joblib").transform(te[FEATURES_COMPAT])

    cpu = platform.processor() or platform.machine()
    py_ver = platform.python_version()
    res = {
        "hardware": f"{cpu} (Python {py_ver}, single-thread n_jobs=1)",
        "rf_full": bench(joblib.load(MODELS / "rf_full.joblib"), X_f),
        "xgb_full": bench(joblib.load(MODELS / "xgb_full.joblib"), X_f),
        "rf_compat": bench(joblib.load(MODELS / "rf_compat.joblib"), X_c),
        "xgb_compat": bench(joblib.load(MODELS / "xgb_compat.joblib"), X_c),
    }
    save_json(res, "inference_benchmark.json")
    for k, v in res.items():
        print(k, "=>", v)


if __name__ == "__main__":
    run()
