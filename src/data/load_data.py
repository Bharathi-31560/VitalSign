"""Dataset loading and summary statistics."""
import pandas as pd
from pathlib import Path

from src.config import DATASET_PATH, ID_COLS, CODE_COLS, TARGET_RAW


class DatasetError(Exception):
    """Raised with a user-friendly message when the dataset cannot be loaded."""


_CACHE = {}


def load_dataset(path=None, use_cache=True):
    """Load the Vital Sign CSV and return a DataFrame.

    "?" and empty cells are read as missing values (NaN). The string "None"
    is kept as a real category (it means "test not performed" in the lab-test
    columns), which is why default NA parsing is switched off.
    Callers must treat the returned frame as read-only (it is cached).
    """
    path = Path(path) if path else DATASET_PATH
    if not path.exists():
        raise DatasetError(
            f"Dataset file not found: data/{path.name}. "
            "Place your Vital Sign CSV in the 'data' folder and name it vital_sign.csv.")
    key = (str(path), path.stat().st_mtime)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    try:
        df = pd.read_csv(path, na_values=["?", ""], keep_default_na=False, low_memory=False)
    except Exception as exc:  # unreadable / corrupt file
        raise DatasetError(f"The dataset could not be read: {exc}") from exc
    if df.empty:
        raise DatasetError("The dataset file is empty.")
    _CACHE.clear()
    _CACHE[key] = df
    return df


def feature_groups(df):
    """Split columns into ids, numeric, categorical (codes count as categorical)."""
    ids = [c for c in ID_COLS if c in df.columns]
    skip = set(ids) | {TARGET_RAW}
    numeric, categorical = [], []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and c not in CODE_COLS:
            numeric.append(c)
        else:
            categorical.append(c)
    return {"ids": ids, "numeric": numeric, "categorical": categorical,
            "target": TARGET_RAW if TARGET_RAW in df.columns else None}


def dataset_summary(df, path=None):
    """Return a dict of descriptive statistics used by dashboard & dataset page."""
    path = Path(path) if path else DATASET_PATH
    groups = feature_groups(df)
    missing = df.isna().sum()
    n = len(df)
    columns = []
    for c in df.columns:
        non_null = df[c].dropna()
        columns.append({
            "name": c,
            "dtype": "text" if not pd.api.types.is_numeric_dtype(df[c]) else str(df[c].dtype),
            "non_null": int(n - missing[c]),
            "missing": int(missing[c]),
            "missing_pct": round(100 * missing[c] / n, 2),
            "unique": int(df[c].nunique(dropna=True)),
            "example": str(non_null.iloc[0]) if len(non_null) else "-",
        })
    target_counts = df[TARGET_RAW].value_counts().to_dict() if groups["target"] else {}
    return {
        "name": path.name,
        "size_mb": round(path.stat().st_size / 1_048_576, 1) if path.exists() else None,
        "rows": int(n),
        "columns": int(df.shape[1]),
        "n_features": int(df.shape[1] - len(groups["ids"]) - (1 if groups["target"] else 0)),
        "numeric": groups["numeric"],
        "categorical": groups["categorical"],
        "ids": groups["ids"],
        "target": groups["target"],
        "missing_cells": int(missing.sum()),
        "missing_cols": {c: int(v) for c, v in missing.items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_patients": int(df["patient_nbr"].duplicated().sum()) if "patient_nbr" in df else 0,
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "n_target_classes": len(target_counts),
        "column_table": columns,
    }
