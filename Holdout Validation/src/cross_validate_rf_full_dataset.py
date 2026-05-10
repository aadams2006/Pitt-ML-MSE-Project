"""Run stratified k-fold cross-validation for the optimized Random Forest on the full dataset."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold


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


def build_results_dir() -> Path:
    """Create the timestamped results directory for the CV run."""
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    run_dir = results_dir / f"stratified_kfold_rf_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def fold_metrics_dict(fold: int, test_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Build a consistent fold-metrics record."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "fold": fold,
        "n_test_samples": int(len(test_indices)),
        "test_index_start": int(test_indices.min()),
        "test_index_end": int(test_indices.max()),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_plots(
    fold_metrics: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    feature_importance_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a summary figure for the cross-validation run."""
    residuals = oof_predictions["actual"] - oof_predictions["predicted"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Optimized Random Forest - Stratified 5-Fold Cross-Validation", fontsize=16)

    axes[0, 0].scatter(oof_predictions["actual"], oof_predictions["predicted"], alpha=0.45)
    axes[0, 0].plot(
        [oof_predictions["actual"].min(), oof_predictions["actual"].max()],
        [oof_predictions["actual"].min(), oof_predictions["actual"].max()],
        "r--",
        lw=2,
    )
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Out-of-Fold Predictions vs Actual")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(fold_metrics["fold"], fold_metrics["R2"], color="#4C78A8")
    axes[0, 1].set_xlabel("Fold")
    axes[0, 1].set_ylabel("R2")
    axes[0, 1].set_title("Fold R2 Scores")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 0].barh(
        feature_importance_summary["feature"],
        feature_importance_summary["importance_mean"],
        xerr=feature_importance_summary["importance_std"],
        color="#59A14F",
    )
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Mean Importance")
    axes[1, 0].set_title("Feature Importances Across Folds")
    axes[1, 0].grid(True, axis="x", alpha=0.3)

    axes[1, 1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Out-of-Fold Residual Distribution")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run stratified k-fold CV on the full 3000-sample dataset and save artifacts."""
    df = pipeline.load_full_data()
    X, y = pipeline.split_features_targets(df)

    valid_idx = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)
    y_flat = np.asarray(y).flatten()

    strata, strata_count = make_stratification_labels(y_flat, n_splits=N_SPLITS, max_bins=MAX_STRATA_BINS)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model_params = dict(pipeline.MODEL_PARAMS)
    model_params["n_jobs"] = 1

    feature_cols, target_cols = pipeline.get_all_features(df)
    fold_metrics: list[dict] = []
    oof_frames: list[pd.DataFrame] = []
    feature_importance_rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, strata), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_flat[train_idx]
        y_test = y_flat[test_idx]

        model = pipeline.build_model(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics.append(fold_metrics_dict(fold, test_idx, y_test, y_pred))
        oof_frames.append(
            pd.DataFrame(
                {
                    "row_index": test_idx,
                    "fold": fold,
                    "actual": y_test,
                    "predicted": y_pred,
                    "residual": y_test - y_pred,
                }
            )
        )

        for feature_name, importance in zip(feature_cols, model.feature_importances_):
            feature_importance_rows.append(
                {
                    "fold": fold,
                    "feature": feature_name,
                    "importance": float(importance),
                }
            )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    oof_predictions_df = pd.concat(oof_frames, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    feature_importances_df = pd.DataFrame(feature_importance_rows)
    feature_importance_summary = (
        feature_importances_df.groupby("feature", as_index=False).agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    overall_mse = mean_squared_error(oof_predictions_df["actual"], oof_predictions_df["predicted"])
    overall_summary = {
        "model": "Random Forest (Optimized)",
        "dataset": "Holdout Validation/Data/synthetic_data_improved.csv",
        "validation_method": "StratifiedKFold on regression target bins",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "n_samples": int(len(oof_predictions_df)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "target": target_cols[0],
        "strata_bins_used": int(strata_count),
        "metrics": {
            "R2_mean": float(fold_metrics_df["R2"].mean()),
            "R2_std": float(fold_metrics_df["R2"].std(ddof=1)),
            "RMSE_mean": float(fold_metrics_df["RMSE"].mean()),
            "RMSE_std": float(fold_metrics_df["RMSE"].std(ddof=1)),
            "MAE_mean": float(fold_metrics_df["MAE"].mean()),
            "MAE_std": float(fold_metrics_df["MAE"].std(ddof=1)),
            "MSE_mean": float(fold_metrics_df["MSE"].mean()),
            "MSE_std": float(fold_metrics_df["MSE"].std(ddof=1)),
            "R2_out_of_fold": float(r2_score(oof_predictions_df["actual"], oof_predictions_df["predicted"])),
            "RMSE_out_of_fold": float(np.sqrt(overall_mse)),
            "MAE_out_of_fold": float(
                mean_absolute_error(oof_predictions_df["actual"], oof_predictions_df["predicted"])
            ),
            "MSE_out_of_fold": float(overall_mse),
        },
        "hyperparameters": model_params,
        "training_date": datetime.now().isoformat(),
    }

    run_dir = build_results_dir()
    fold_metrics_df.to_csv(run_dir / "fold_metrics.csv", index=False)
    fold_metrics_df.to_json(run_dir / "fold_metrics.json", orient="records", indent=2)
    oof_predictions_df.to_csv(run_dir / "out_of_fold_predictions.csv", index=False)
    feature_importances_df.to_csv(run_dir / "feature_importances_by_fold.csv", index=False)
    feature_importance_summary.to_csv(run_dir / "feature_importances_summary.csv", index=False)

    with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    save_plots(
        fold_metrics=fold_metrics_df,
        oof_predictions=oof_predictions_df,
        feature_importance_summary=feature_importance_summary,
        output_path=run_dir / "stratified_kfold_rf_optimized_summary.png",
    )

    readme_lines = [
        "# Stratified K-Fold Cross-Validation Results",
        "",
        f"Training date: {overall_summary['training_date']}",
        "",
        "## Configuration",
        "",
        f"- Dataset: `{overall_summary['dataset']}`",
        f"- Samples: {overall_summary['n_samples']}",
        f"- Features: {overall_summary['n_features']}",
        f"- Target: `{overall_summary['target']}`",
        f"- Folds: {overall_summary['n_splits']}",
        f"- Stratification bins: {overall_summary['strata_bins_used']}",
        "- Stratification method: quantile bins over the continuous target",
        "- Model source: `Holdout Validation/models/pipeline_rf_optimized.py`",
        "",
        "## Aggregate Performance",
        "",
        f"- Mean R2: {overall_summary['metrics']['R2_mean']:.6f} +/- {overall_summary['metrics']['R2_std']:.6f}",
        f"- Mean RMSE: {overall_summary['metrics']['RMSE_mean']:.6f} +/- {overall_summary['metrics']['RMSE_std']:.6f}",
        f"- Mean MAE: {overall_summary['metrics']['MAE_mean']:.6f} +/- {overall_summary['metrics']['MAE_std']:.6f}",
        f"- Out-of-fold R2: {overall_summary['metrics']['R2_out_of_fold']:.6f}",
        f"- Out-of-fold RMSE: {overall_summary['metrics']['RMSE_out_of_fold']:.6f}",
        "",
        "## Outputs",
        "",
        "- `summary_metrics.json`",
        "- `fold_metrics.csv` and `fold_metrics.json`",
        "- `out_of_fold_predictions.csv`",
        "- `feature_importances_by_fold.csv`",
        "- `feature_importances_summary.csv`",
        "- `stratified_kfold_rf_optimized_summary.png`",
    ]
    (run_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Results saved to: {run_dir}")
    print(f"Mean R2: {overall_summary['metrics']['R2_mean']:.6f} +/- {overall_summary['metrics']['R2_std']:.6f}")
    print(f"Out-of-fold R2: {overall_summary['metrics']['R2_out_of_fold']:.6f}")
    print(
        f"Mean RMSE: {overall_summary['metrics']['RMSE_mean']:.6f} +/- "
        f"{overall_summary['metrics']['RMSE_std']:.6f}"
    )


if __name__ == "__main__":
    main()
