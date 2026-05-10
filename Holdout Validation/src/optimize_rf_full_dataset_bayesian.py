"""Bayesian hyperparameter optimization for the holdout-validated Random Forest on the full dataset."""

from __future__ import annotations

import importlib.util
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def load_module(name: str, path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_dir = Path(__file__).resolve().parent.parent
pipeline = load_module("pipeline_rf_optimized", base_dir / "models" / "pipeline_rf_optimized.py")

N_SPLITS = 5
MAX_STRATA_BINS = 10
RANDOM_STATE = 42
N_ITER = 32


def build_results_dir() -> Path:
    """Create the timestamped results directory for the optimization run."""
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    run_dir = results_dir / f"rf_bayesian_optimization_full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(exist_ok=True)
    return run_dir


def make_stratification_labels(y: np.ndarray, n_splits: int, max_bins: int) -> tuple[pd.Series, int]:
    """Create regression strata by quantile-binning the target."""
    y_series = pd.Series(y, name="target")

    for bins in range(max_bins, 1, -1):
        strata = pd.qcut(y_series, q=bins, labels=False, duplicates="drop")
        counts = strata.value_counts()
        if counts.empty:
            continue
        if counts.min() >= n_splits and counts.size >= 2:
            return strata.astype(int), counts.size

    raise ValueError("Unable to create valid strata for StratifiedKFold.")


def build_search_space() -> dict:
    """Define the Bayesian search space around the validated RF configuration."""
    return {
        "n_estimators": Integer(100, 450),
        "max_depth": Integer(8, 30),
        "min_samples_split": Integer(2, 12),
        "min_samples_leaf": Integer(1, 6),
        "max_features": Categorical(["sqrt", "log2", None]),
        "bootstrap": Categorical([True, False]),
    }


def make_metric_dict(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a consistent metric bundle."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "label": name,
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
    }


def save_plot(cv_results_df: pd.DataFrame, output_path: Path) -> None:
    """Save a compact optimization summary plot."""
    ranked_df = cv_results_df.sort_values("iteration").reset_index(drop=True)
    best_so_far = ranked_df["mean_test_r2"].cummax()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bayesian Optimization of Random Forest on Full Dataset", fontsize=16)

    axes[0].plot(ranked_df["iteration"], ranked_df["mean_test_r2"], marker="o", alpha=0.7, label="Candidate")
    axes[0].plot(ranked_df["iteration"], best_so_far, color="red", lw=2, label="Best so far")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean CV R2")
    axes[0].set_title("Search Progress")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(ranked_df["mean_test_r2"], ranked_df["mean_fit_time"], alpha=0.7)
    axes[1].set_xlabel("Mean CV R2")
    axes[1].set_ylabel("Mean Fit Time (s)")
    axes[1].set_title("Fit Time vs Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Optimize the holdout-validated RF on the full dataset and save outputs."""
    df = pipeline.load_full_data()
    X, y = pipeline.split_features_targets(df)

    valid_idx = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)
    y_flat = np.asarray(y).flatten()

    strata, strata_count = make_stratification_labels(y_flat, n_splits=N_SPLITS, max_bins=MAX_STRATA_BINS)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(X, strata))

    baseline_params = dict(pipeline.MODEL_PARAMS)
    baseline_params["n_jobs"] = 1
    search_base_params = {
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }

    optimizer = BayesSearchCV(
        estimator=pipeline.build_model(**search_base_params),
        search_spaces=build_search_space(),
        n_iter=N_ITER,
        scoring="r2",
        cv=cv_splits,
        n_jobs=1,
        random_state=RANDOM_STATE,
        return_train_score=True,
        verbose=0,
        refit=True,
    )
    optimizer.fit(X, y_flat, groups=None)

    best_params = dict(optimizer.best_params_)
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = 1

    baseline_estimator = pipeline.build_model(**baseline_params)
    optimized_estimator = pipeline.build_model(**best_params)

    baseline_oof_pred = cross_val_predict(baseline_estimator, X, y_flat, cv=cv_splits, n_jobs=1)
    optimized_oof_pred = cross_val_predict(optimized_estimator, X, y_flat, cv=cv_splits, n_jobs=1)

    baseline_metrics = make_metric_dict("Baseline RF", y_flat, baseline_oof_pred)
    optimized_metrics = make_metric_dict("Optimized RF", y_flat, optimized_oof_pred)

    feature_cols, target_cols = pipeline.get_all_features(df)
    optimizer.best_estimator_.fit(X, y_flat)
    full_pred = optimizer.best_estimator_.predict(X)
    full_fit_metrics = make_metric_dict("Optimized RF Full-Fit", y_flat, full_pred)

    cv_results_df = pd.DataFrame(optimizer.cv_results_)
    cv_results_df = cv_results_df.rename(
        columns={
            "mean_test_score": "mean_test_r2",
            "std_test_score": "std_test_r2",
            "mean_train_score": "mean_train_r2",
            "std_train_score": "std_train_r2",
        }
    )
    cv_results_df["iteration"] = np.arange(1, len(cv_results_df) + 1)
    cv_results_df = cv_results_df.sort_values("rank_test_score").reset_index(drop=True)

    best_estimator = optimizer.best_estimator_
    feature_importances_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": best_estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    oof_predictions_df = pd.DataFrame(
        {
            "actual": y_flat,
            "baseline_prediction": baseline_oof_pred,
            "optimized_prediction": optimized_oof_pred,
            "optimized_residual": y_flat - optimized_oof_pred,
        }
    )

    parameter_columns = [col for col in cv_results_df.columns if col.startswith("param_")]
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
    ].copy()

    run_dir = build_results_dir()
    search_results_df.to_csv(run_dir / "bayes_search_results.csv", index=False)
    feature_importances_df.to_csv(run_dir / "feature_importances.csv", index=False)
    oof_predictions_df.to_csv(run_dir / "out_of_fold_predictions.csv", index=False)

    with open(run_dir / "best_estimator.pkl", "wb") as f:
        pickle.dump(best_estimator, f)

    summary_payload = {
        "dataset": "Holdout Validation/Data/synthetic_data_improved.csv",
        "n_samples": int(len(X)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "target": target_cols[0],
        "validation_method": "StratifiedKFold on regression target bins",
        "n_splits": N_SPLITS,
        "strata_bins_used": int(strata_count),
        "optimizer": "BayesSearchCV",
        "n_iter": N_ITER,
        "scoring": "r2",
        "baseline_params": baseline_params,
        "best_params": best_params,
        "best_cv_r2": float(optimizer.best_score_),
        "baseline_oof_metrics": baseline_metrics,
        "optimized_oof_metrics": optimized_metrics,
        "full_fit_metrics": full_fit_metrics,
        "training_date": datetime.now().isoformat(),
    }
    with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    readme_lines = [
        "# Bayesian RF Hyperparameter Optimization",
        "",
        f"Training date: {summary_payload['training_date']}",
        "",
        "## Configuration",
        "",
        f"- Dataset: `{summary_payload['dataset']}`",
        f"- Samples: {summary_payload['n_samples']}",
        f"- Features: {summary_payload['n_features']}",
        f"- CV folds: {summary_payload['n_splits']}",
        f"- Stratification bins: {summary_payload['strata_bins_used']}",
        f"- Bayesian iterations: {summary_payload['n_iter']}",
        "",
        "## Performance",
        "",
        f"- Baseline RF OOF R2: {baseline_metrics['R2']:.6f}",
        f"- Optimized RF OOF R2: {optimized_metrics['R2']:.6f}",
        f"- Optimized RF best CV R2: {summary_payload['best_cv_r2']:.6f}",
        f"- Full-fit R2 on all 3000 samples: {full_fit_metrics['R2']:.6f}",
        "",
        "## Best Parameters",
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
            "- `best_estimator.pkl`",
            "- `optimization_summary.png`",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    save_plot(search_results_df, run_dir / "optimization_summary.png")

    print(f"Results saved to: {run_dir}")
    print(f"Baseline OOF R2: {baseline_metrics['R2']:.6f}")
    print(f"Optimized OOF R2: {optimized_metrics['R2']:.6f}")
    print(f"Best CV R2: {summary_payload['best_cv_r2']:.6f}")
    print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()
