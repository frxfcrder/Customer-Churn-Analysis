from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_COLS, TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

PREDICT_COLS = [col for col in FEATURE_COLS if col != TARGET]


def save_model(model, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def predict_churn(model, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        missing = [col for col in PREDICT_COLS if col not in X.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        X = X[PREDICT_COLS]
    return np.asarray(model.predict(X))


def predict_churn_proba(model, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        missing = [col for col in PREDICT_COLS if col not in X.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        X = X[PREDICT_COLS]
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"{type(model).__name__} has no predict_proba")
    return np.asarray(model.predict_proba(X)[:, 1])


def predict_single(model, record: dict) -> float:
    return float(predict_churn(model, pd.DataFrame([record]))[0])