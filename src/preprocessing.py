from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.2
RANDOM_STATE = 42
IMPUTE_STRATEGY = "median"


def check_missing(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    return missing[missing > 0]


def impute_missing(
    df: pd.DataFrame,
    strategy: str = IMPUTE_STRATEGY,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = list(df.columns) if columns is None else list(columns)
    imputer = SimpleImputer(strategy=strategy)
    df = df.copy()
    df[columns] = imputer.fit_transform(df[columns])
    return df


def split_train_test(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray,
           pd.Series | np.ndarray, pd.Series | np.ndarray]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


def fit_scaler(X_train: pd.DataFrame | np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(
    scaler: StandardScaler, X: pd.DataFrame | np.ndarray
) -> np.ndarray:
    return scaler.transform(X)


def scale_train_test(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray, StandardScaler]:
    """Fit on train only -> no leakage from the test fold.

    DataFrames keep their columns and index so downstream code can keep
    using feature names instead of positional arrays.
    """
    scaler = fit_scaler(X_train)
    X_train_s = transform_features(scaler, X_train)
    X_test_s = transform_features(scaler, X_test)

    if isinstance(X_train, pd.DataFrame):
        X_train_s = pd.DataFrame(
            X_train_s, columns=X_train.columns, index=X_train.index
        )
    if isinstance(X_test, pd.DataFrame):
        X_test_s = pd.DataFrame(
            X_test_s, columns=X_test.columns, index=X_test.index
        )
    return X_train_s, X_test_s, scaler
