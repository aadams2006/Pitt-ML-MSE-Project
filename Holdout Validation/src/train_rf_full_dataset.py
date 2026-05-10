"""Train the holdout-optimized Random Forest on the full 3000-sample dataset."""

from __future__ import annotations

import importlib.util
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_module(name: str, path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_dir = Path(__file__).resolve().parent.parent
pipeline = load_module("pipeline_rf_optimized", base_dir / "models" / "pipeline_rf_optimized.py")


def build_results_dir() -> Path:
    """Create the timestamped results directory for the full-dataset retrain."""
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    run_dir = results_dir / f"full_dataset_rf_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(exist_ok=True)
    return run_dir


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, oob_r2: float | None) -> dict:
    """Compute summary metrics for the fitted model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    metrics = {
        "model": "Random Forest (Optimized Full Dataset)",
        "dataset": "Holdout Validation/Data/synthetic_data_improved.csv",
        "n_samples": int(len(y_true)),
        "n_features": None,
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": mae,
        "R2_training": r2,
        "OOB_R2": None if oob_r2 is None else float(oob_r2),
        "training_date": datetime.now().isoformat(),
        "hyperparameters": dict(pipeline.MODEL_PARAMS),
    }
    return metrics


def save_plot(y_true: np.ndarray, y_pred: np.ndarray, feature_importances: pd.DataFrame, output_path: Path) -> None:
    """Save a compact training diagnostics plot."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Random Forest (Optimized) - Full 3000-Sample Training", fontsize=16)

    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Training Predictions vs Actual")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Training Residuals")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].barh(feature_importances["feature"], feature_importances["importance"], color="#4C78A8")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Importance")
    axes[1, 0].set_title("Feature Importances")
    axes[1, 0].grid(True, axis="x", alpha=0.3)

    axes[1, 1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Residual Distribution")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Fit the optimized Random Forest on the full dataset and save artifacts."""
    df = pipeline.load_full_data()
    X, y = pipeline.split_features_targets(df)

    valid_idx = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)
    y_flat = np.asarray(y).flatten()

    model = pipeline.build_model(oob_score=True)
    model.fit(X, y_flat)
    y_pred = model.predict(X)

    feature_cols, target_cols = pipeline.get_all_features(df)
    metrics = compute_metrics(y_flat, y_pred, getattr(model, "oob_score_", None))
    metrics["n_features"] = len(feature_cols)
    metrics["features"] = feature_cols
    metrics["target"] = target_cols[0]

    run_dir = build_results_dir()

    predictions_df = pd.DataFrame(
        {
            "actual": y_flat,
            "predicted": y_pred,
            "residual": y_flat - y_pred,
        }
    )
    predictions_df.to_csv(run_dir / "training_predictions.csv", index=False)

    feature_importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importances.to_csv(run_dir / "feature_importances.csv", index=False)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "random_forest_optimized_full_dataset.pkl", "wb") as f:
        pickle.dump(model, f)

    save_plot(
        y_true=y_flat,
        y_pred=y_pred,
        feature_importances=feature_importances,
        output_path=run_dir / "random_forest_optimized_full_dataset_training.png",
    )

    readme_lines = [
        "# Full-Dataset Random Forest Retrain",
        "",
        f"Training date: {metrics['training_date']}",
        "",
        "## Configuration",
        "",
        f"- Dataset: `{metrics['dataset']}`",
        f"- Samples: {metrics['n_samples']}",
        f"- Features: {metrics['n_features']}",
        f"- Target: `{metrics['target']}`",
        "- Model source: `Holdout Validation/models/pipeline_rf_optimized.py`",
        "",
        "## Metrics",
        "",
        f"- Training R2: {metrics['R2_training']:.6f}",
        f"- OOB R2: {metrics['OOB_R2']:.6f}" if metrics["OOB_R2"] is not None else "- OOB R2: unavailable",
        f"- RMSE: {metrics['RMSE']:.6f}",
        f"- MAE: {metrics['MAE']:.6f}",
        f"- MSE: {metrics['MSE']:.6f}",
        "",
        "## Outputs",
        "",
        "- `metrics.json`",
        "- `training_predictions.csv`",
        "- `feature_importances.csv`",
        "- `random_forest_optimized_full_dataset.pkl`",
        "- `random_forest_optimized_full_dataset_training.png`",
    ]
    (run_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Results saved to: {run_dir}")
    print(f"Training R2: {metrics['R2_training']:.6f}")
    if metrics["OOB_R2"] is not None:
        print(f"OOB R2: {metrics['OOB_R2']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")


if __name__ == "__main__":
    main()
