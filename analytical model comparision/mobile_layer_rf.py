"""Shared reconstruction helpers for the optimized mobile-layer Random Forest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


COMPARISON_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = COMPARISON_ROOT.parent
TRAINING_RESULTS_ROOT = PROJECT_ROOT / "Holdout Validation" / "results"

TOTAL_COLUMN = "Total Thickness (nm)"
BONDED_COLUMN = "Bonded Thickness (nm)"
MOBILE_TARGET_COLUMN = "Mobile Layer (nm)"


def find_latest_training_summary() -> Path:
    """Locate the newest committed mobile-layer Bayesian optimization summary."""
    candidates = sorted(
        TRAINING_RESULTS_ROOT.glob(
            "rf_bayesian_optimization_mobile_layer_*/summary_metrics.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "No mobile-layer optimization summary found. Run "
            "`python \"Holdout Validation/src/optimize_rf_mobile_layer_bayesian.py\"` first."
        )
    return candidates[-1]


def derive_mobile_target(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Derive mobile thickness and clip tiny negative measurement differences."""
    missing = [column for column in (TOTAL_COLUMN, BONDED_COLUMN) if column not in df]
    if missing:
        raise ValueError(f"Experimental data are missing mobile-target columns: {missing}")
    raw_mobile = df[TOTAL_COLUMN] - df[BONDED_COLUMN]
    audit = {
        "definition": f"{MOBILE_TARGET_COLUMN} = {TOTAL_COLUMN} - {BONDED_COLUMN}",
        "negative_rows_clipped_to_zero": int((raw_mobile < 0.0).sum()),
        "minimum_before_clipping_nm": float(raw_mobile.min()),
    }
    return raw_mobile.clip(lower=0.0).rename(MOBILE_TARGET_COLUMN), audit


def rebuild_mobile_model() -> tuple[RandomForestRegressor, dict, Path]:
    """Rebuild and fit the optimized RF from its documented best parameters."""
    summary_path = find_latest_training_summary()
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    dataset_path = PROJECT_ROOT / summary["dataset"]
    training_df = pd.read_csv(dataset_path)
    training_df.columns = training_df.columns.str.strip()
    features = summary["features"]
    raw_mobile = training_df[TOTAL_COLUMN] - training_df[BONDED_COLUMN]
    valid_mask = ~(training_df[features].isna().any(axis=1) | raw_mobile.isna())
    valid_mask &= raw_mobile >= 0.0

    actual_count = int(valid_mask.sum())
    expected_count = int(summary["data_audit"]["training_rows"])
    if actual_count != expected_count:
        raise ValueError(
            "Mobile-layer training-row count no longer matches optimization metadata: "
            f"expected {expected_count}, found {actual_count}."
        )

    model = RandomForestRegressor(**summary["best_params"])
    model.fit(training_df.loc[valid_mask, features], raw_mobile.loc[valid_mask])
    return model, summary, summary_path
