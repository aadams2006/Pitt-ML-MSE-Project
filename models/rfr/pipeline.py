"""Placeholder pipeline for Random Forest Regression (RFR)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "withdrawal_speed",
    "dwell_time_s",
    "substrate_type",
    "pdms_concentration",
    "solvent_type",
    "etc_feature",
]

TARGET_COLUMNS = [
    "total_film_thickness",
    "bonded_film_thickness",
]


def load_placeholder_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load the shared placeholder dataset for quick experiments."""
    if data_path is None:
        data_path = Path(__file__).resolve().parents[2] / "data" / "placeholder_film_data.csv"
    return pd.read_csv(data_path)


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into feature and target matrices."""
    features = df[FEATURE_COLUMNS].copy()
    targets = df[TARGET_COLUMNS].copy()
    return features, targets
