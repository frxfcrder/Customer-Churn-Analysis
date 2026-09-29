from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .features import TARGET, select_features, select_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
CSV_FILENAME = "telecom_churn.csv"
KAGGLE_DATASET = "barun2104/telecom-churn"


def download_dataset(dest_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    
    import kagglehub

    raw_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    source = raw_path / CSV_FILENAME
    if not source.exists():
        raise FileNotFoundError(f"{CSV_FILENAME} not found in {raw_path}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / CSV_FILENAME
    if not target.exists():
        shutil.copyfile(source, target)
    return target


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    if path is not None:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
    else:
        csv_path = DEFAULT_DATA_DIR / CSV_FILENAME
        if not csv_path.exists():
            csv_path = download_dataset()

    return pd.read_csv(csv_path)


def split_features_target(
    df: pd.DataFrame, target: str = TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    return select_features(df), select_target(df, target)