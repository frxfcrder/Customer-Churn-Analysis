from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
DEFAULT_MAX_ITER = 1000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

RF_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
}
RF_CV_FOLDS = 5
RF_SCORING = "roc_auc"

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}


def build_models() -> dict:
    return {
        "Logistic": LogisticRegression(
            max_iter=DEFAULT_MAX_ITER, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "HistGB": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "MLP": MLPClassifier(max_iter=DEFAULT_MAX_ITER, random_state=RANDOM_STATE),
    }


def build_xgboost_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", StandardScaler()),
            ("model", XGBClassifier(**XGB_PARAMS)),
        ]
    )


def train_models(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> dict:
    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def tune_random_forest(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    param_grid: dict | None = None,
) -> GridSearchCV:
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid or RF_PARAM_GRID,
        cv=RF_CV_FOLDS,
        scoring=RF_SCORING,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def train_all(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> dict:
    models = train_models(X_train, y_train)
    grid = tune_random_forest(X_train, y_train)
    models["RF (Tuned)"] = grid.best_estimator_
    xgb_pipeline = build_xgboost_pipeline()
    xgb_pipeline.fit(X_train, y_train)
    models["XGBoost"] = xgb_pipeline
    return models


def save_model(model, name: str, models_dir: str | Path = DEFAULT_MODELS_DIR) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(path: str | Path):
    return joblib.load(path)
