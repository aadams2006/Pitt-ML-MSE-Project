"""Bayesian optimization of Random Forest regression for mobile-layer thickness.

The measured mobile layer is reconstructed as total thickness minus bonded
thickness. Rows for which that difference is negative are physically invalid
synthetic samples and are excluded from this target-specific training run. The
source dataset itself is never modified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature engineering v1"
    / "data FE-V1"
    / "synthetic_data_improved.csv"
)
DATASET_REPO_PATH = "models/feature engineering v1/data FE-V1/synthetic_data_improved.csv"
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

TOTAL_COLUMN = "Total Thickness (nm)"
BONDED_COLUMN = "Bonded Thickness (nm)"
TARGET_COLUMN = "Mobile Layer (nm)"
FEATURE_COLUMNS = [
    "Polarity (XLogP3)",
    "Viscosity (cP)",
    "Boiling Point (K)",
    "Surface Tension (mN/m)",
    "Concentration (g/mL)",
    "Uncoated Layer (nm)",
    TOTAL_COLUMN,
]

N_SPLITS = 5
MAX_STRATA_BINS = 10
RANDOM_STATE = 42
DEFAULT_N_ITER = 32
BASELINE_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help=f"Bayesian-search iterations (default: {DEFAULT_N_ITER}).",
    )
    return parser.parse_args()


def load_mobile_layer_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load features and derive a physically valid mobile-layer target."""
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()

    required = set(FEATURE_COLUMNS + [BONDED_COLUMN])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    raw_mobile = df[TOTAL_COLUMN] - df[BONDED_COLUMN]
    complete_mask = ~(df[FEATURE_COLUMNS].isna().any(axis=1) | raw_mobile.isna())
    nonnegative_mask = raw_mobile >= 0.0
    valid_mask = complete_mask & nonnegative_mask

    X = df.loc[valid_mask, FEATURE_COLUMNS].reset_index(drop=True)
    y = raw_mobile.loc[valid_mask].rename(TARGET_COLUMN).reset_index(drop=True)
    audit = {
        "input_rows": int(len(df)),
        "rows_with_missing_values": int((~complete_mask).sum()),
        "negative_mobile_rows_excluded": int((complete_mask & ~nonnegative_mask).sum()),
        "training_rows": int(valid_mask.sum()),
        "target_definition": f"{TARGET_COLUMN} = {TOTAL_COLUMN} - {BONDED_COLUMN}",
        "invalid_row_policy": "Exclude rows with derived mobile thickness < 0; source data remain unchanged.",
    }
    return X, y, audit


def make_stratification_labels(
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    max_bins: int = MAX_STRATA_BINS,
) -> tuple[pd.Series, int]:
    """Create regression strata by quantile-binning the mobile target."""
    y_series = pd.Series(y, name="target")
    for bins in range(max_bins, 1, -1):
        strata = pd.qcut(y_series, q=bins, labels=False, duplicates="drop")
        counts = strata.value_counts()
        if not counts.empty and counts.min() >= n_splits and counts.size >= 2:
            return strata.astype(int), int(counts.size)
    raise ValueError("Unable to create valid target strata for StratifiedKFold.")


def build_search_space() -> dict:
    """Use the same RF search domain as the bonded-thickness ceiling run."""
    return {
        "n_estimators": Integer(100, 450),
        "max_depth": Integer(8, 30),
        "min_samples_split": Integer(2, 12),
        "min_samples_leaf": Integer(1, 6),
        "max_features": Categorical(["sqrt", "log2", None]),
        "bootstrap": Categorical([True, False]),
    }


def make_model(params: dict) -> RandomForestRegressor:
    return RandomForestRegressor(**params)


def make_metric_dict(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "label": label,
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
    }


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_plot(cv_results_df: pd.DataFrame, output_path: Path) -> None:
    ranked_df = cv_results_df.sort_values("iteration").reset_index(drop=True)
    best_so_far = ranked_df["mean_test_r2"].cummax()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bayesian RF Optimization for Mobile-Layer Thickness", fontsize=16)
    axes[0].plot(
        ranked_df["iteration"],
        ranked_df["mean_test_r2"],
        marker="o",
        alpha=0.7,
        label="Candidate",
    )
    axes[0].plot(ranked_df["iteration"], best_so_far, color="red", lw=2, label="Best so far")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean CV $R^2$")
    axes[0].set_title("Search Progress")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(ranked_df["mean_test_r2"], ranked_df["mean_fit_time"], alpha=0.7)
    axes[1].set_xlabel("Mean CV $R^2$")
    axes[1].set_ylabel("Mean Fit Time (s)")
    axes[1].set_title("Fit Time vs. Score")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_results_dir() -> Path:
    RESULTS_ROOT.mkdir(exist_ok=True)
    run_dir = RESULTS_ROOT / (
        "rf_bayesian_optimization_mobile_layer_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(exist_ok=False)
    return run_dir


def main() -> None:
    args = parse_args()
    if args.n_iter < 1:
        raise ValueError("--n-iter must be at least 1")

    X, y, data_audit = load_mobile_layer_data()
    y_array = y.to_numpy(dtype=float)
    strata, strata_count = make_stratification_labels(y_array)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(X, strata))

    optimizer = BayesSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        search_spaces=build_search_space(),
        n_iter=args.n_iter,
        scoring="r2",
        cv=cv_splits,
        n_jobs=1,
        random_state=RANDOM_STATE,
        return_train_score=True,
        verbose=0,
        refit=True,
    )
    optimizer.fit(X, y_array)

    best_params = dict(optimizer.best_params_)
    best_params.update({"random_state": RANDOM_STATE, "n_jobs": 1})
    baseline_oof_pred = cross_val_predict(
        make_model(BASELINE_PARAMS), X, y_array, cv=cv_splits, n_jobs=1
    )
    optimized_oof_pred = cross_val_predict(
        make_model(best_params), X, y_array, cv=cv_splits, n_jobs=1
    )
    baseline_metrics = make_metric_dict("Baseline RF", y_array, baseline_oof_pred)
    optimized_metrics = make_metric_dict("Bayesian-Optimized RF", y_array, optimized_oof_pred)

    full_model = make_model(best_params)
    full_model.fit(X, y_array)
    full_pred = full_model.predict(X)
    full_fit_metrics = make_metric_dict("Bayesian-Optimized RF Full-Fit", y_array, full_pred)

    cv_results_df = pd.DataFrame(optimizer.cv_results_).rename(
        columns={
            "mean_test_score": "mean_test_r2",
            "std_test_score": "std_test_r2",
            "mean_train_score": "mean_train_r2",
            "std_train_score": "std_train_r2",
        }
    )
    cv_results_df["iteration"] = np.arange(1, len(cv_results_df) + 1)
    parameter_columns = [column for column in cv_results_df if column.startswith("param_")]
    search_results_df = cv_results_df[
        [
            "iteration",
            "rank_test_score",
            "mean_test_r2",
            "std_test_r2",
            "mean_train_r2",
            "std_train_r2",
            "mean_fit_time",
            "std_fit_time",
        ]
        + parameter_columns
    ].sort_values("rank_test_score")

    feature_importances_df = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": full_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    oof_predictions_df = pd.DataFrame(
        {
            "actual_mobile_layer_nm": y_array,
            "baseline_prediction_nm": baseline_oof_pred,
            "optimized_prediction_nm": optimized_oof_pred,
            "optimized_residual_nm": y_array - optimized_oof_pred,
        }
    )

    run_dir = build_results_dir()
    search_results_df.to_csv(run_dir / "bayes_search_results.csv", index=False)
    feature_importances_df.to_csv(run_dir / "feature_importances.csv", index=False)
    oof_predictions_df.to_csv(run_dir / "out_of_fold_predictions.csv", index=False)

    summary_payload = to_json_safe(
        {
            "dataset": DATASET_REPO_PATH,
            "data_audit": data_audit,
            "n_features": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "target_source_columns": [TOTAL_COLUMN, BONDED_COLUMN],
            "validation_method": "StratifiedKFold on mobile-target quantile bins",
            "n_splits": N_SPLITS,
            "strata_bins_used": strata_count,
            "optimizer": "BayesSearchCV",
            "n_iter": args.n_iter,
            "scoring": "r2",
            "baseline_params": BASELINE_PARAMS,
            "best_params": best_params,
            "best_cv_r2": optimizer.best_score_,
            "baseline_oof_metrics": baseline_metrics,
            "optimized_oof_metrics": optimized_metrics,
            "full_fit_metrics": full_fit_metrics,
            "training_date": datetime.now().isoformat(),
            "serialized_estimator_committed": False,
            "reconstruction_note": (
                "Comparison runners rebuild and fit the estimator from best_params and the "
                "documented cleaned training rows, avoiding another large binary artifact."
            ),
        }
    )
    with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    readme_lines = [
        "# Bayesian RF Optimization: Mobile Layer",
        "",
        f"Training date: {summary_payload['training_date']}",
        "",
        "## Target and data policy",
        "",
        f"- Target: `{data_audit['target_definition']}`",
        f"- Source dataset: `{DATASET_REPO_PATH}`",
        f"- Input rows: {data_audit['input_rows']}",
        f"- Negative mobile-layer rows excluded: {data_audit['negative_mobile_rows_excluded']}",
        f"- Rows used for training: {data_audit['training_rows']}",
        "- The source CSV was not modified.",
        "",
        "## Validation and performance",
        "",
        f"- Bayesian iterations: {args.n_iter}",
        f"- Stratified folds: {N_SPLITS}",
        f"- Baseline RF OOF R2: {baseline_metrics['R2']:.6f}",
        f"- Optimized RF OOF R2: {optimized_metrics['R2']:.6f}",
        f"- Optimized RF OOF RMSE: {optimized_metrics['RMSE']:.6f} nm",
        f"- Best search CV R2: {summary_payload['best_cv_r2']:.6f}",
        "",
        "## Best parameters",
        "",
    ]
    for key, value in best_params.items():
        readme_lines.append(f"- `{key}`: `{value}`")
    readme_lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `summary_metrics.json`",
            "- `bayes_search_results.csv`",
            "- `out_of_fold_predictions.csv`",
            "- `feature_importances.csv`",
            "- `optimization_summary.png`",
            "",
            "The estimator is reproducibly rebuilt from `best_params` by the solvent comparison runners; no duplicate large model binary is committed.",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    save_plot(cv_results_df, run_dir / "optimization_summary.png")

    print(f"Results saved to: {run_dir}")
    print(f"Training rows: {data_audit['training_rows']} / {data_audit['input_rows']}")
    print(f"Negative mobile rows excluded: {data_audit['negative_mobile_rows_excluded']}")
    print(f"Baseline OOF R2: {baseline_metrics['R2']:.6f}")
    print(f"Optimized OOF R2: {optimized_metrics['R2']:.6f}")
    print(f"Optimized OOF RMSE: {optimized_metrics['RMSE']:.6f} nm")
    print(f"Best params: {to_json_safe(best_params)}")


if __name__ == "__main__":
    main()
