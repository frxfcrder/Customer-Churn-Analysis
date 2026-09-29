from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.features import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET  
N_SAMPLES = 300
RANDOM_STATE = 42


@pytest.fixture
def churn_df() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame(
        rng.normal(size=(N_SAMPLES, len(NUMERICAL_COLS))),
        columns=NUMERICAL_COLS,
    )
    for col in CATEGORICAL_COLS:
        df[col] = rng.integers(0, 2, size=N_SAMPLES)
    df[TARGET] = rng.integers(0, 2, size=N_SAMPLES)
    return df


@pytest.fixture
def df_with_missing(churn_df: pd.DataFrame) -> pd.DataFrame:
    df = churn_df.copy()
    df.loc[:9, "DayMins"] = np.nan
    return df
