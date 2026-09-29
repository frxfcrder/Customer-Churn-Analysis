from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import (
    RANDOM_STATE,
    TEST_SIZE,
    check_missing,
    impute_missing,
    scale_train_test,
    split_train_test,
)


def test_check_missing_no_nulls(churn_df: pd.DataFrame):
    assert check_missing(churn_df).empty


def test_check_missing_reports_only_null_columns(df_with_missing: pd.DataFrame):
    missing = check_missing(df_with_missing)
    assert list(missing.index) == ["DayMins"]
    assert int(missing["DayMins"]) == 10


def test_impute_missing_fills_with_median(df_with_missing: pd.DataFrame):
    expected = df_with_missing["DayMins"].median()
    clean = impute_missing(df_with_missing)

    assert check_missing(clean).empty
    assert clean["DayMins"].iloc[:10].eq(expected).all()
    assert np.isnan(df_with_missing["DayMins"]).any()  # original untouched


def test_impute_missing_only_selected_columns(df_with_missing: pd.DataFrame):
    clean = impute_missing(df_with_missing, columns=["DayMins"])
    assert check_missing(clean).empty
    assert list(clean.columns) == list(df_with_missing.columns)


def test_split_train_test_shapes(churn_df: pd.DataFrame):
    X = churn_df.drop(columns=["Churn"])
    y = churn_df["Churn"]
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    n_test = int(np.ceil(len(churn_df) * TEST_SIZE))
    assert len(X_test) == n_test
    assert len(X_train) == len(churn_df) - n_test
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_split_train_test_is_stratified(churn_df: pd.DataFrame):
    X = churn_df.drop(columns=["Churn"])
    y = churn_df["Churn"]
    _, _, y_train, y_test = split_train_test(X, y)

    rate = y.mean()
    assert abs(y_train.mean() - rate) < 0.05
    assert abs(y_test.mean() - rate) < 0.05


def test_split_train_test_without_stratify(churn_df: pd.DataFrame):
    X = churn_df.drop(columns=["Churn"])
    y = churn_df["Churn"]
    X_train, X_test, _, _ = split_train_test(X, y, stratify=False)
    assert len(X_train) + len(X_test) == len(churn_df)


def test_split_train_test_is_reproducible(churn_df: pd.DataFrame):
    X = churn_df.drop(columns=["Churn"])
    y = churn_df["Churn"]
    first = split_train_test(X, y)
    second = split_train_test(X, y, random_state=RANDOM_STATE)
    assert first[0].index.tolist() == second[0].index.tolist()


def test_scale_train_test_fits_on_train_only(churn_df: pd.DataFrame):
    X = churn_df.drop(columns=["Churn"])
    y = churn_df["Churn"]
    X_train, X_test, _, _ = split_train_test(X, y)
    X_train_s, X_test_s, scaler = scale_train_test(X_train, X_test)

    assert isinstance(X_train_s, pd.DataFrame)
    assert list(X_train_s.columns) == list(X_train.columns)
    assert X_train_s.shape == X_train.shape
    assert X_test_s.shape == X_test.shape
    np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy(), atol=1e-9)
    assert X_train_s.mean().abs().max() < 1e-9
    # test fold scaled with train statistics, so its mean is not exactly 0
    assert X_test_s.mean().abs().max() > 1e-9
