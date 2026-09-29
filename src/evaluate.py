from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
DEFAULT_LABELS = ("No Churn", "Churn")
CHURN_PROBA_SLICE = 1


def positive_proba(model, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    return model.predict_proba(X)[:, CHURN_PROBA_SLICE]


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    metrics["ROC-AUC"] = (
        roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    )
    return metrics


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X)
    return compute_metrics(y, y_pred, positive_proba(model, X))


def evaluate_models(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    round_to: int = 4,
) -> pd.DataFrame:
    results = [
        {"Model": name, **evaluate_model(model, X, y)}
        for name, model in models.items()
    ]
    return pd.DataFrame(results).set_index("Model").round(round_to)


def best_model(results: pd.DataFrame, metric: str = "F1") -> str:
    if metric not in results.columns:
        raise KeyError(f"{metric} not in {list(results.columns)}")
    return results[metric].idxmax()


def confusion_matrix_df(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    labels: tuple[str, str] = DEFAULT_LABELS,
) -> pd.DataFrame:
    cm = confusion_matrix(y, model.predict(X))
    return pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )


def roc_curve_data(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    y_prob = positive_proba(model, X)
    if y_prob is None:
        raise ValueError(f"{type(model).__name__} has no predict_proba")
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    return pd.DataFrame(
        {"fpr": fpr, "tpr": tpr, "threshold": thresholds,
         "auc": roc_auc_score(y, y_prob)}
    )


def precision_recall_curve_data(
    model, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    y_prob = positive_proba(model, X)
    if y_prob is None:
        raise ValueError(f"{type(model).__name__} has no predict_proba")
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    return pd.DataFrame(
        {"precision": precision[:-1], "recall": recall[:-1],
         "threshold": thresholds, "ap": average_precision_score(y, y_prob)}
    )


def classification_report_str(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    labels: tuple[str, str] = DEFAULT_LABELS,
) -> str:
    return classification_report(
        y, model.predict(X), target_names=list(labels), zero_division=0
    )
