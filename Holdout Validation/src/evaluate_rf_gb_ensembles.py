"""Evaluate RF, GB, and RF+GB ensembles with stratified k-fold cross-validation."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold


def load_module(name: str, path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_dir = Path(__file__).resolve().parent.parent
rf_pipeline = load_module("pipeline_rf_optimized", base_dir / "models" / "pipeline_rf_optimized.py")
gb_pipeline = load_module("pipeline_gb_optimized", base_dir / "models" / "pipeline_gb_optimized.py")

N_SPLITS = 5
INNER_SPLITS = 5
MAX_STRATA_BINS = 10
RANDOM_STATE = 42
BLEND_WEIGHTS = [round(weight, 1) for weight in np.linspace(0.0, 1.0, 11)]


def build_results_dir() -> Path:
    """Create the timestamped results directory for the ensemble evaluation."""
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    run_dir = results_dir / f"ensemble_rf_gb_stratified_kfold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def fit_base_models(X_train: pd.DataFrame, y_train: np.ndarray):
    """Fit the optimized RF and GB models on one training fold."""
    rf_params = dict(rf_pipeline.MODEL_PARAMS)
    rf_params["n_jobs"] = 1
    gb_params = dict(gb_pipeline.MODEL_PARAMS)

    rf_model = rf_pipeline.build_model(**rf_params)
    gb_model = gb_pipeline.build_model(**gb_params)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    return rf_model, gb_model, rf_params, gb_params


def make_metric_row(
    model_name: str,
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: dict | None = None,
) -> dict:
    """Create a metric record for one model on one outer fold."""
    mse = mean_squared_error(y_true, y_pred)
    row = {
        "model": model_name,
        "fold": fold,
        "n_test_samples": int(len(y_true)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
    if extra:
        row.update(extra)
    return row


def run_inner_blend_selection(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Select the RF blend weight with inner stratified CV."""
    inner_strata, _ = make_stratification_labels(y_train, n_splits=INNER_SPLITS, max_bins=MAX_STRATA_BINS)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    weight_rows: list[dict] = []
    for weight in BLEND_WEIGHTS:
        rf_weight = float(weight)
        gb_weight = float(1.0 - weight)
        oof_pred = np.zeros(len(y_train), dtype=float)

        for inner_train_idx, inner_valid_idx in inner_cv.split(X_train, inner_strata):
            X_inner_train = X_train.iloc[inner_train_idx]
            X_inner_valid = X_train.iloc[inner_valid_idx]
            y_inner_train = y_train[inner_train_idx]

            rf_model, gb_model, _, _ = fit_base_models(X_inner_train, y_inner_train)
            rf_pred = rf_model.predict(X_inner_valid)
            gb_pred = gb_model.predict(X_inner_valid)
            oof_pred[inner_valid_idx] = rf_weight * rf_pred + gb_weight * gb_pred

        weight_rows.append(
            {
                "rf_weight": rf_weight,
                "gb_weight": gb_weight,
                "R2": float(r2_score(y_train, oof_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_train, oof_pred))),
                "MAE": float(mean_absolute_error(y_train, oof_pred)),
            }
        )

    weight_df = pd.DataFrame(weight_rows).sort_values(["R2", "RMSE"], ascending=[False, True]).reset_index(drop=True)
    best_weight = float(weight_df.loc[0, "rf_weight"])
    return best_weight, weight_df


def run_inner_stacking(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[Ridge, pd.DataFrame]:
    """Fit a Ridge meta-model using cross-fitted base predictions."""
    inner_strata, _ = make_stratification_labels(y_train, n_splits=INNER_SPLITS, max_bins=MAX_STRATA_BINS)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    meta_features = np.zeros((len(y_train), 2), dtype=float)
    for inner_train_idx, inner_valid_idx in inner_cv.split(X_train, inner_strata):
        X_inner_train = X_train.iloc[inner_train_idx]
        X_inner_valid = X_train.iloc[inner_valid_idx]
        y_inner_train = y_train[inner_train_idx]

        rf_model, gb_model, _, _ = fit_base_models(X_inner_train, y_inner_train)
        meta_features[inner_valid_idx, 0] = rf_model.predict(X_inner_valid)
        meta_features[inner_valid_idx, 1] = gb_model.predict(X_inner_valid)

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features, y_train)
    coefficients = pd.DataFrame(
        [
            {
                "meta_feature": "rf_prediction",
                "coefficient": float(meta_model.coef_[0]),
            },
            {
                "meta_feature": "gb_prediction",
                "coefficient": float(meta_model.coef_[1]),
            },
            {
                "meta_feature": "intercept",
                "coefficient": float(meta_model.intercept_),
            },
        ]
    )
    return meta_model, coefficients


def save_plots(
    summary_df: pd.DataFrame,
    oof_summary_df: pd.DataFrame,
    feature_importance_summary_df: pd.DataFrame,
    residual_target_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a compact overview plot for the ensemble run."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("RF + GB Ensemble Comparison on Full Dataset", fontsize=16)

    axes[0, 0].bar(summary_df["model"], summary_df["R2_mean"], yerr=summary_df["R2_std"], color="#4C78A8")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].set_ylabel("Mean Fold R2")
    axes[0, 0].set_title("Cross-Validation Performance")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].bar(oof_summary_df["model"], oof_summary_df["R2_out_of_fold"], color="#59A14F")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].set_ylabel("Out-of-Fold R2")
    axes[0, 1].set_title("Overall OOF Performance")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    top_features = feature_importance_summary_df[feature_importance_summary_df["model"] == "Adaptive Blend"].head(7)
    axes[1, 0].barh(top_features["feature"], top_features["importance_mean"], color="#E15759")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Mean Effective Importance")
    axes[1, 0].set_title("Adaptive Blend Feature Importance")
    axes[1, 0].grid(True, axis="x", alpha=0.3)

    axes[1, 1].plot(
        residual_target_df["target_bin"],
        residual_target_df["mae"],
        marker="o",
        color="#F28E2B",
    )
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].set_ylabel("MAE")
    axes[1, 1].set_title("Best Model Residuals by Target Decile")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Evaluate base and ensemble models on the full dataset."""
    df = rf_pipeline.load_full_data()
    X, y = rf_pipeline.split_features_targets(df)

    valid_idx = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)
    y_flat = np.asarray(y).flatten()

    strata, strata_count = make_stratification_labels(y_flat, n_splits=N_SPLITS, max_bins=MAX_STRATA_BINS)
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    feature_cols, target_cols = rf_pipeline.get_all_features(df)

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    feature_importance_rows: list[dict] = []
    blend_selection_rows: list[dict] = []
    stacking_coeff_rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, strata), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_flat[train_idx]
        y_test = y_flat[test_idx]

        rf_model, gb_model, _, _ = fit_base_models(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)

        best_rf_weight, blend_search_df = run_inner_blend_selection(X_train, y_train)
        best_gb_weight = 1.0 - best_rf_weight
        fixed_blend_pred = 0.5 * rf_pred + 0.5 * gb_pred
        adaptive_blend_pred = best_rf_weight * rf_pred + best_gb_weight * gb_pred

        meta_model, coefficients_df = run_inner_stacking(X_train, y_train)
        stacked_pred = meta_model.predict(np.column_stack([rf_pred, gb_pred]))

        metric_rows.extend(
            [
                make_metric_row("Random Forest", fold, y_test, rf_pred),
                make_metric_row("Gradient Boosting", fold, y_test, gb_pred),
                make_metric_row("Fixed Blend 50/50", fold, y_test, fixed_blend_pred),
                make_metric_row(
                    "Adaptive Blend",
                    fold,
                    y_test,
                    adaptive_blend_pred,
                    {"rf_weight": best_rf_weight, "gb_weight": best_gb_weight},
                ),
                make_metric_row("Stacked Ridge", fold, y_test, stacked_pred),
            ]
        )

        fold_prediction_frames = {
            "Random Forest": rf_pred,
            "Gradient Boosting": gb_pred,
            "Fixed Blend 50/50": fixed_blend_pred,
            "Adaptive Blend": adaptive_blend_pred,
            "Stacked Ridge": stacked_pred,
        }
        for model_name, preds in fold_prediction_frames.items():
            prediction_rows.extend(
                [
                    {
                        "model": model_name,
                        "fold": fold,
                        "row_index": int(row_idx),
                        "actual": float(actual),
                        "predicted": float(predicted),
                        "residual": float(actual - predicted),
                    }
                    for row_idx, actual, predicted in zip(test_idx, y_test, preds)
                ]
            )

        for feature_name, rf_importance, gb_importance in zip(
            feature_cols,
            rf_model.feature_importances_,
            gb_model.feature_importances_,
        ):
            feature_importance_rows.extend(
                [
                    {
                        "model": "Random Forest",
                        "fold": fold,
                        "feature": feature_name,
                        "importance": float(rf_importance),
                    },
                    {
                        "model": "Gradient Boosting",
                        "fold": fold,
                        "feature": feature_name,
                        "importance": float(gb_importance),
                    },
                    {
                        "model": "Adaptive Blend",
                        "fold": fold,
                        "feature": feature_name,
                        "importance": float(best_rf_weight * rf_importance + best_gb_weight * gb_importance),
                    },
                ]
            )

        best_inner_row = blend_search_df.iloc[0].to_dict()
        blend_selection_rows.append(
            {
                "fold": fold,
                "selected_rf_weight": best_rf_weight,
                "selected_gb_weight": best_gb_weight,
                "inner_cv_R2": float(best_inner_row["R2"]),
                "inner_cv_RMSE": float(best_inner_row["RMSE"]),
                "inner_cv_MAE": float(best_inner_row["MAE"]),
            }
        )

        for _, row in coefficients_df.iterrows():
            stacking_coeff_rows.append(
                {
                    "fold": fold,
                    "meta_feature": row["meta_feature"],
                    "coefficient": row["coefficient"],
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows).sort_values(["model", "row_index"]).reset_index(drop=True)
    feature_importances_df = pd.DataFrame(feature_importance_rows)
    blend_selection_df = pd.DataFrame(blend_selection_rows)
    stacking_coeff_df = pd.DataFrame(stacking_coeff_rows)

    summary_df = (
        metrics_df.groupby("model", as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
        )
        .sort_values(["R2_mean", "RMSE_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )

    oof_summary_rows: list[dict] = []
    for model_name, group in predictions_df.groupby("model"):
        actual = group["actual"].to_numpy()
        predicted = group["predicted"].to_numpy()
        mse = mean_squared_error(actual, predicted)
        oof_summary_rows.append(
            {
                "model": model_name,
                "R2_out_of_fold": float(r2_score(actual, predicted)),
                "RMSE_out_of_fold": float(np.sqrt(mse)),
                "MAE_out_of_fold": float(mean_absolute_error(actual, predicted)),
                "MSE_out_of_fold": float(mse),
            }
        )
    oof_summary_df = pd.DataFrame(oof_summary_rows).sort_values(
        ["R2_out_of_fold", "RMSE_out_of_fold"],
        ascending=[False, True],
    ).reset_index(drop=True)

    feature_importance_summary_df = (
        feature_importances_df.groupby(["model", "feature"], as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
        )
        .sort_values(["model", "importance_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    best_model = str(oof_summary_df.iloc[0]["model"])
    best_predictions_df = predictions_df[predictions_df["model"] == best_model].copy().reset_index(drop=True)
    best_predictions_df["target_bin"] = pd.qcut(best_predictions_df["actual"], q=10, duplicates="drop")
    residual_target_df = (
        best_predictions_df.groupby("target_bin", observed=False)
        .agg(
            mean_residual=("residual", "mean"),
            mae=("residual", lambda x: float(np.mean(np.abs(x)))),
            rmse=("residual", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            count=("residual", "size"),
        )
        .reset_index()
    )
    residual_target_df["target_bin"] = residual_target_df["target_bin"].astype(str)

    residual_feature_rows: list[dict] = []
    for feature_name in feature_cols:
        feature_values = X.loc[best_predictions_df["row_index"], feature_name].reset_index(drop=True)
        residuals = best_predictions_df["residual"].reset_index(drop=True)
        abs_residuals = residuals.abs()
        quartiles = pd.qcut(feature_values, q=4, duplicates="drop")
        quartile_mae = abs_residuals.groupby(quartiles, observed=False).mean()

        residual_feature_rows.append(
            {
                "feature": feature_name,
                "feature_skew": float(pd.Series(feature_values).skew()),
                "residual_corr": float(pd.Series(feature_values).corr(residuals)),
                "abs_residual_corr": float(pd.Series(feature_values).corr(abs_residuals)),
                "mae_low_quartile": float(quartile_mae.iloc[0]),
                "mae_high_quartile": float(quartile_mae.iloc[-1]),
                "mae_high_low_ratio": float(quartile_mae.iloc[-1] / quartile_mae.iloc[0]) if quartile_mae.iloc[0] else np.nan,
            }
        )
    residual_feature_df = pd.DataFrame(residual_feature_rows).sort_values(
        ["abs_residual_corr", "mae_high_low_ratio"],
        ascending=[False, False],
    ).reset_index(drop=True)

    suggestion_rows: list[dict] = []
    for _, row in residual_feature_df.iterrows():
        suggestion = None
        rationale_parts = []

        if abs(row["residual_corr"]) >= 0.08:
            rationale_parts.append(f"signed residual correlation {row['residual_corr']:.3f}")
        if abs(row["abs_residual_corr"]) >= 0.08:
            rationale_parts.append(f"absolute residual correlation {row['abs_residual_corr']:.3f}")
        if row["mae_high_low_ratio"] >= 1.25:
            rationale_parts.append(f"high/low quartile MAE ratio {row['mae_high_low_ratio']:.2f}")

        if not rationale_parts:
            continue

        if row["feature_skew"] >= 1.0:
            suggestion = "Test log1p transform or quantile binning for this feature."
        elif "Thickness" in row["feature"]:
            suggestion = "Test thickness interaction terms such as total-to-uncoated ratio or difference."
        elif row["feature"] in {"Polarity (XLogP3)", "Surface Tension (mN/m)", "Viscosity (cP)"}:
            suggestion = "Test pairwise interactions among physicochemical variables."
        else:
            suggestion = "Test spline or piecewise transformation for this feature."

        suggestion_rows.append(
            {
                "feature": row["feature"],
                "diagnostic_summary": "; ".join(rationale_parts),
                "suggested_next_step": suggestion,
            }
        )
    suggestion_df = pd.DataFrame(suggestion_rows)

    run_dir = build_results_dir()
    metrics_df.to_csv(run_dir / "fold_metrics_by_model.csv", index=False)
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)
    oof_summary_df.to_csv(run_dir / "out_of_fold_summary.csv", index=False)
    predictions_df.to_csv(run_dir / "out_of_fold_predictions_by_model.csv", index=False)
    feature_importances_df.to_csv(run_dir / "feature_importances_by_fold.csv", index=False)
    feature_importance_summary_df.to_csv(run_dir / "feature_importances_summary.csv", index=False)
    blend_selection_df.to_csv(run_dir / "adaptive_blend_weights_by_fold.csv", index=False)
    stacking_coeff_df.to_csv(run_dir / "stacking_coefficients_by_fold.csv", index=False)
    residual_target_df.to_csv(run_dir / f"residuals_by_target_bin_{best_model.replace(' ', '_').lower()}.csv", index=False)
    residual_feature_df.to_csv(run_dir / f"residual_feature_diagnostics_{best_model.replace(' ', '_').lower()}.csv", index=False)
    suggestion_df.to_csv(run_dir / "feature_transformation_candidates.csv", index=False)

    summary_payload = {
        "dataset": "Holdout Validation/Data/synthetic_data_improved.csv",
        "validation_method": "StratifiedKFold on regression target bins",
        "n_splits": N_SPLITS,
        "inner_splits_for_ensembles": INNER_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "strata_bins_used": int(strata_count),
        "n_samples": int(len(X)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "target": target_cols[0],
        "rf_hyperparameters": {**rf_pipeline.MODEL_PARAMS, "n_jobs": 1},
        "gb_hyperparameters": dict(gb_pipeline.MODEL_PARAMS),
        "blend_weight_grid": BLEND_WEIGHTS,
        "best_model_by_oof_r2": best_model,
        "summary_metrics": summary_df.to_dict(orient="records"),
        "out_of_fold_metrics": oof_summary_df.to_dict(orient="records"),
        "training_date": datetime.now().isoformat(),
    }
    with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    save_plots(
        summary_df=summary_df,
        oof_summary_df=oof_summary_df,
        feature_importance_summary_df=feature_importance_summary_df,
        residual_target_df=residual_target_df,
        output_path=run_dir / "ensemble_comparison_summary.png",
    )

    readme_lines = [
        "# RF + GB Ensemble Evaluation",
        "",
        f"Training date: {summary_payload['training_date']}",
        "",
        "## Configuration",
        "",
        "- Dataset: `Holdout Validation/Data/synthetic_data_improved.csv`",
        f"- Samples: {summary_payload['n_samples']}",
        f"- Features: {summary_payload['n_features']}",
        f"- Folds: {summary_payload['n_splits']}",
        f"- Inner folds for blend/stacking selection: {summary_payload['inner_splits_for_ensembles']}",
        "- Models evaluated: Random Forest, Gradient Boosting, 50/50 blend, adaptive blend, stacked ridge",
        "",
        "## Best Model",
        "",
        f"- Best by out-of-fold R2: {best_model}",
        "",
        "## Out-of-Fold R2",
        "",
    ]
    for _, row in oof_summary_df.iterrows():
        readme_lines.append(f"- {row['model']}: {row['R2_out_of_fold']:.6f}")

    readme_lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `summary_metrics.json` and `summary_metrics.csv`",
            "- `fold_metrics_by_model.csv`",
            "- `out_of_fold_summary.csv`",
            "- `out_of_fold_predictions_by_model.csv`",
            "- `feature_importances_by_fold.csv` and `feature_importances_summary.csv`",
            "- `adaptive_blend_weights_by_fold.csv`",
            "- `stacking_coefficients_by_fold.csv`",
            "- `feature_transformation_candidates.csv`",
            "- `ensemble_comparison_summary.png`",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Results saved to: {run_dir}")
    for _, row in oof_summary_df.iterrows():
        print(f"{row['model']}: OOF R2={row['R2_out_of_fold']:.6f}, RMSE={row['RMSE_out_of_fold']:.6f}")
    print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()
