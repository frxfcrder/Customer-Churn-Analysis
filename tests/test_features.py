from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERICAL_COLS,
    TARGET,
    feature_importance,
    get_feature_names,
    select_features,
    select_target,
)


def test_target_is_not_a_feature():
    assert TARGET == "Churn"
    assert TARGET not in FEATURE_COLS
    assert TARGET not in CATEGORICAL_COLS
    assert TARGET not in NUMERICAL_COLS


def test_feature_columns_partition():
    assert set(FEATURE_COLS) == set(CATEGORICAL_COLS) | set(NUMERICAL_COLS)
    assert set(CATEGORICAL_COLS).isdisjoint(NUMERICAL_COLS)
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS)) == 10
    assert FEATURE_COLS[1:3] == CATEGORICAL_COLS  # same order as the CSV


def test_select_features_excludes_target(churn_df: pd.DataFrame):
    X = select_features(churn_df)
    assert TARGET not in X.columns
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(churn_df)


def test_select_features_returns_copy(churn_df: pd.DataFrame):
    X = select_features(churn_df)
    X.iloc[0, 0] = 999.0
    assert churn_df.iloc[0, 0] != 999.0


def test_select_features_missing_column_raises(churn_df: pd.DataFrame):
    with pytest.raises(KeyError):
        select_features(churn_df.drop(columns=["DayMins"]))


def test_select_target(churn_df: pd.DataFrame):
    y = select_target(churn_df)
    assert y.name == TARGET
    assert len(y) == len(churn_df)
    assert set(y.unique()) <= {0, 1}


def test_select_target_missing_raises(churn_df: pd.DataFrame):
    with pytest.raises(KeyError):
        select_target(churn_df.drop(columns=[TARGET]))


def test_get_feature_names_strips_prefixes():
    class FakePreprocessor:
        def get_feature_names_out(self):
            return np.array(["cat__DataPlan", "remainder__DayMins"])

    assert get_feature_names(FakePreprocessor()) == ["DataPlan", "DayMins"]


def test_feature_importance_from_pipeline(churn_df: pd.DataFrame):
    from src.train import build_xgboost_pipeline

    X = select_features(churn_df)
    y = select_target(churn_df)
    pipeline = build_xgboost_pipeline()
    pipeline.fit(X, y)

    importances = feature_importance(pipeline, top_n=5)

    assert isinstance(importances, pd.Series)
    assert len(importances) == 5
    assert set(importances.index) <= set(FEATURE_COLS)
    assert importances.is_monotonic_decreasing
