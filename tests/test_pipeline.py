from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import split_features_target
from src.evaluate import METRIC_NAMES, best_model, evaluate_models
from src.predict import (
    load_model,
    predict_churn,
    predict_churn_proba,
    predict_single,
    save_model,
)
from src.preprocessing import scale_train_test, split_train_test
from src.train import train_all

EXPECTED_MODELS = [
    "Logistic", "Decision Tree", "Random Forest", "AdaBoost",
    "HistGB", "MLP", "RF (Tuned)", "XGBoost",
]


@pytest.fixture
def trained(churn_df: pd.DataFrame):
    X, y = split_features_target(churn_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    models = train_all(X_train, y_train)
    return models, X_train, X_test, y_train, y_test


def test_train_all_returns_every_model(trained):
    models, *_ = trained
    assert list(models) == EXPECTED_MODELS
    assert all(hasattr(m, "predict") for m in models.values())


def test_evaluate_models_table(trained):
    models, _, X_test, _, y_test = trained
    results = evaluate_models(models, X_test, y_test)

    assert list(results.index) == EXPECTED_MODELS
    assert list(results.columns) == list(METRIC_NAMES)
    assert results.notna().all().all()
    assert ((results >= 0) & (results <= 1)).all().all()
    assert best_model(results) in EXPECTED_MODELS
    assert best_model(results, metric="ROC-AUC") in EXPECTED_MODELS


def test_evaluate_models_unknown_metric_raises(trained):
    models, _, X_test, _, y_test = trained
    results = evaluate_models(models, X_test, y_test)
    with pytest.raises(KeyError):
        best_model(results, metric="MCC")


def test_predict_churn_returns_labels(trained):
    models, _, X_test, _, _ = trained
    preds = predict_churn(models["XGBoost"], X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)) <= {0, 1}


def test_predict_churn_probabilities(trained):
    models, _, X_test, _, _ = trained
    proba = predict_churn_proba(models["Random Forest"], X_test)

    assert len(proba) == len(X_test)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_churn_missing_column_raises(trained):
    models, _, X_test, _, _ = trained
    with pytest.raises(KeyError):
        predict_churn(models["XGBoost"], X_test.drop(columns=["DayMins"]))


def test_predict_single(trained):
    models, _, X_test, _, _ = trained
    record = X_test.iloc[0].to_dict()
    prediction = predict_single(models["Logistic"], record)

    assert isinstance(prediction, float)
    assert prediction in (0.0, 1.0)


def test_ignores_target_column_if_present(trained):
    models, _, X_test, _, y_test = trained
    with_target = X_test.assign(Churn=y_test.to_numpy())
    preds = predict_churn(models["XGBoost"], with_target)

    assert len(preds) == len(X_test)


def test_save_and_load_model(trained, tmp_path):
    models, *_ = trained
    path = save_model(models["XGBoost"], tmp_path / "xgboost.joblib")
    assert path.exists()

    loaded = load_model(path)
    _, _, X_test, _, _ = trained
    np.testing.assert_array_equal(
        predict_churn(models["XGBoost"], X_test),
        predict_churn(loaded, X_test),
    )


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")


def test_scaled_features_flow(trained):
    models, X_train, X_test, y_train, y_test = trained
    X_train_s, X_test_s, _ = scale_train_test(X_train, X_test)
    models = train_all(X_train_s, y_train)
    results = evaluate_models(models, X_test_s, y_test)

    assert list(results.index) == EXPECTED_MODELS
    assert results.notna().all().all()
