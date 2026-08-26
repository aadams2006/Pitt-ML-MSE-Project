"""Run side-by-side evaluation of the best ML model and analytical models."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from analytical_models import (
    DEFAULT_DENSITY_KG_M3,
    DEFAULT_DWELL_TIME_S,
    DEFAULT_FILM_WIDTH_M,
    DEFAULT_WITHDRAWAL_SPEED_MM_S,
    EXPERIMENT_SOLUTE,
    EXPERIMENT_SOLVENT,
    HEXANE_RELATIVE_EVAPORATION_BUAC,
    HEXANE_RELATIVE_EVAPORATION_SOURCE,
    get_bonded_layer_models,
    get_mobile_layer_models,
)

COMPARISON_ROOT = Path(__file__).resolve().parents[2]
if str(COMPARISON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARISON_ROOT))

from mobile_layer_rf import (  # noqa: E402
    MOBILE_TARGET_COLUMN,
    derive_mobile_target,
    rebuild_mobile_model,
)


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTAL_DATA_PATH = (
    PROJECT_ROOT / "models" / "feature engineering v1" / "data FE-V1" / "agg.data.xlsx"
)

BEST_MODEL_PATH = ARTIFACTS_DIR / "best_estimator.pkl"
BEST_MODEL_METADATA_PATH = ARTIFACTS_DIR / "summary_metrics.json"

TARGET_COLUMN = "Bonded Thickness (nm)"
HEXANE_PROPERTIES = {
    "Polarity (XLogP3)": 3.9,
    "Viscosity (cP)": 0.377,
    "Boiling Point (K)": 342.039,
    "Surface Tension (mN/m)": 17.89,
}


def load_best_model() -> tuple[object, dict]:
    """Load the saved best estimator and its metadata."""
    with open(BEST_MODEL_PATH, "rb") as handle:
        model = pickle.load(handle)
    with open(BEST_MODEL_METADATA_PATH, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return model, metadata


def load_experimental_data() -> pd.DataFrame:
    """Load the original experimental dataset used for comparison."""
    df = pd.read_excel(EXPERIMENTAL_DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


def build_ml_feature_frame(
    experimental_df: pd.DataFrame,
    feature_order: list[str],
) -> pd.DataFrame:
    """Augment the experimental rows so they match the best RF feature schema."""
    ml_df = experimental_df.copy()
    for column, value in HEXANE_PROPERTIES.items():
        if column not in ml_df.columns:
            ml_df[column] = value
    missing = [column for column in feature_order if column not in ml_df.columns]
    if missing:
        raise ValueError(f"Missing features required by the saved ML model: {missing}")
    return ml_df[feature_order].copy()


def metric_bundle(y_true: pd.Series, y_pred: pd.Series, model_name: str, model_type: str) -> dict:
    """Compute a consistent set of regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model_name,
        "type": model_type,
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
    }


def save_scatter_plot(predictions_df: pd.DataFrame, output_path: Path) -> None:
    """Save a simple actual-vs-predicted comparison plot."""
    model_columns = [
        column
        for column in predictions_df.columns
        if column not in {"actual_bonded_thickness", "row_id"}
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    actual = predictions_df["actual_bonded_thickness"]

    for column in model_columns:
        ax.scatter(actual, predictions_df[column], alpha=0.7, label=column)

    lower = min(actual.min(), *(predictions_df[column].min() for column in model_columns))
    upper = max(actual.max(), *(predictions_df[column].max() for column in model_columns))
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Actual Bonded Thickness (nm)")
    ax.set_ylabel("Predicted Bonded Thickness (nm)")
    ax.set_title("Experimental Comparison: ML vs Analytical Models")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_mobile_scatter_plot(predictions_df: pd.DataFrame, output_path: Path) -> None:
    """Save the mobile-target comparison without mixing target definitions."""
    model_columns = [
        column
        for column in predictions_df.columns
        if column not in {"actual_mobile_layer", "row_id"}
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    actual = predictions_df["actual_mobile_layer"]
    for column in model_columns:
        ax.scatter(actual, predictions_df[column], alpha=0.7, label=column)
    lower = min(actual.min(), *(predictions_df[column].min() for column in model_columns))
    upper = max(actual.max(), *(predictions_df[column].max() for column in model_columns))
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Actual Mobile Layer (nm)")
    ax.set_ylabel("Predicted Mobile Layer (nm)")
    ax.set_title("Hexane Mobile Layer: Bayesian RF vs Landau--Levich")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run the comparison and write results to disk."""
    RESULTS_DIR.mkdir(exist_ok=True)
    run_dir = RESULTS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(exist_ok=True)

    best_model, model_metadata = load_best_model()
    experimental_df = load_experimental_data()
    y_true = experimental_df[TARGET_COLUMN].copy()

    feature_order = model_metadata["features"]
    ml_features = build_ml_feature_frame(experimental_df, feature_order)
    ml_pred = pd.Series(best_model.predict(ml_features), index=experimental_df.index, name="Bayesian Optimized RF")

    predictions_df = pd.DataFrame(
        {
            "row_id": np.arange(1, len(experimental_df) + 1),
            "actual_bonded_thickness": y_true,
            "Bayesian Optimized RF": ml_pred,
        }
    )
    metrics = [metric_bundle(y_true, ml_pred, "Bayesian Optimized RF", "machine_learning")]

    analytical_models = get_bonded_layer_models()
    symbolic_models: list[dict] = []
    for analytical_model in analytical_models:
        if analytical_model.requires_effective_e:
            symbolic_models.append(
                {
                    "model": analytical_model.name,
                    "reason": "Requires effective evaporation rate E for numeric evaluation.",
                    "symbolic_expression": analytical_model.symbolic_expression,
                }
            )
            continue
        analytical_pred = analytical_model.predict(experimental_df)
        predictions_df[analytical_model.name] = analytical_pred
        metrics.append(
            metric_bundle(
                y_true,
                analytical_pred,
                analytical_model.name,
                "analytical",
            )
        )

    metrics_df = pd.DataFrame(metrics).sort_values(["type", "RMSE", "MAE"]).reset_index(drop=True)

    predictions_df.to_csv(run_dir / "comparison_predictions.csv", index=False)
    metrics_df.to_csv(run_dir / "comparison_metrics.csv", index=False)

    mobile_model, mobile_model_metadata, mobile_summary_path = rebuild_mobile_model()
    mobile_feature_order = mobile_model_metadata["features"]
    mobile_features = build_ml_feature_frame(experimental_df, mobile_feature_order)
    mobile_true, mobile_target_audit = derive_mobile_target(experimental_df)
    mobile_ml_pred = pd.Series(
        mobile_model.predict(mobile_features),
        index=experimental_df.index,
        name="Bayesian Optimized RF (Mobile Layer)",
    )
    mobile_predictions_df = pd.DataFrame(
        {
            "row_id": np.arange(1, len(experimental_df) + 1),
            "actual_mobile_layer": mobile_true,
            "Bayesian Optimized RF (Mobile Layer)": mobile_ml_pred,
        }
    )
    mobile_metrics = [
        metric_bundle(
            mobile_true,
            mobile_ml_pred,
            "Bayesian Optimized RF (Mobile Layer)",
            "machine_learning",
        )
    ]
    mobile_analytical_models = get_mobile_layer_models()
    for analytical_model in mobile_analytical_models:
        analytical_pred = analytical_model.predict(experimental_df)
        mobile_predictions_df[analytical_model.name] = analytical_pred
        mobile_metrics.append(
            metric_bundle(
                mobile_true,
                analytical_pred,
                analytical_model.name,
                "analytical",
            )
        )
    mobile_metrics_df = (
        pd.DataFrame(mobile_metrics)
        .sort_values(["type", "RMSE", "MAE"])
        .reset_index(drop=True)
    )
    mobile_predictions_df.to_csv(run_dir / "mobile_layer_predictions.csv", index=False)
    mobile_metrics_df.to_csv(run_dir / "mobile_layer_metrics.csv", index=False)

    comparison_metadata = {
        "created_at": datetime.now().isoformat(),
        "experimental_data_path": str(EXPERIMENTAL_DATA_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "best_model_training_summary": str(BEST_MODEL_METADATA_PATH),
        "target_column": TARGET_COLUMN,
        "bonded_comparison_excludes_landau_levich": True,
        "feature_order": feature_order,
        "mobile_layer_comparison": {
            "target_column": MOBILE_TARGET_COLUMN,
            "target_audit": mobile_target_audit,
            "feature_order": mobile_feature_order,
            "optimization_summary": str(mobile_summary_path),
            "training_data_audit": mobile_model_metadata["data_audit"],
            "models": ["Bayesian Optimized RF (Mobile Layer)"]
            + [model.name for model in mobile_analytical_models],
        },
        "hexane_properties_used_for_ml_only": HEXANE_PROPERTIES,
        "analytical_model_experiment_identity": {
            "solute": EXPERIMENT_SOLUTE,
            "solvent": EXPERIMENT_SOLVENT,
        },
        "hexane_evaporation_reference": {
            "source": HEXANE_RELATIVE_EVAPORATION_SOURCE,
            "relative_evaporation_buac_equals_1": HEXANE_RELATIVE_EVAPORATION_BUAC,
            "note": "Relative evaporation data support that hexane is fast-evaporating, but do not provide the effective model E in m/s.",
        },
        "fixed_experimental_constants_used_by_analytical_models": {
            "dwell_time_s": DEFAULT_DWELL_TIME_S,
            "withdrawal_speed_mm_s": DEFAULT_WITHDRAWAL_SPEED_MM_S,
            "film_width_m": DEFAULT_FILM_WIDTH_M,
            "density_kg_m3": DEFAULT_DENSITY_KG_M3,
        },
        "analytical_models_registered": [model.name for model in analytical_models],
        "symbolic_models_requiring_e": symbolic_models,
        "n_rows_evaluated": int(len(experimental_df)),
    }
    with open(run_dir / "comparison_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(comparison_metadata, handle, indent=2)

    save_scatter_plot(predictions_df, run_dir / "comparison_plot.png")
    save_mobile_scatter_plot(
        mobile_predictions_df,
        run_dir / "mobile_layer_comparison_plot.png",
    )

    readme_lines = [
        "# Comparison Run",
        "",
        f"Created: {comparison_metadata['created_at']}",
        "",
        "## Inputs",
        "",
        f"- Experimental dataset: `{EXPERIMENTAL_DATA_PATH}`",
        f"- Best model artifact: `{BEST_MODEL_PATH}`",
        f"- Rows evaluated: {comparison_metadata['n_rows_evaluated']}",
        f"- Analytical models are currently configured for `{EXPERIMENT_SOLUTE}` in `{EXPERIMENT_SOLVENT}`.",
        "- Concentration is read directly from `agg.data.xlsx`.",
        "- Dwell time, withdrawal speed, film width, and density are fixed experiment-level constants in the current implementation.",
        f"- Hexane relative evaporation reference: `{HEXANE_RELATIVE_EVAPORATION_SOURCE}`, with `BuAc = 1 -> {HEXANE_RELATIVE_EVAPORATION_BUAC}`.",
        "- That relative evaporation value does not provide the effective model evaporation rate in m/s.",
        "- The density used in the Landau-Levich term is the coating-solution density, currently approximated by hexane for the dilute PDMS + hexane bath.",
        "",
        "## Bonded-Thickness Models Included",
        "",
        "- `Bayesian Optimized RF`",
    ]
    if analytical_models:
        for analytical_model in analytical_models:
            if analytical_model.requires_effective_e:
                readme_lines.append(f"- `{analytical_model.name}`: symbolic only, requires `E`")
            else:
                readme_lines.append(f"- `{analytical_model.name}`")
    else:
        readme_lines.append("- No analytical formulas registered yet.")
    if symbolic_models:
        readme_lines.extend(
            [
                "",
                "## Symbolic Models",
                "",
            ]
        )
        for symbolic_model in symbolic_models:
            readme_lines.append(
                f"- `{symbolic_model['model']}`: `{symbolic_model['symbolic_expression']}`"
            )
    readme_lines.extend(
        [
            "",
            "Landau--Levich is intentionally excluded from this bonded-thickness comparison.",
            "",
            "## Mobile-Layer Comparison",
            "",
            "- Target: `Mobile Layer (nm) = Total Thickness (nm) - Bonded Thickness (nm)`",
            "- Models: `Bayesian Optimized RF (Mobile Layer)` and `Landau-Levich Mobile Layer`",
            f"- Mobile RF optimization summary: `{mobile_summary_path}`",
            f"- Negative experimental differences clipped to zero: {mobile_target_audit['negative_rows_clipped_to_zero']}",
            "- Landau--Levich wet thickness is converted to dry mobile PDMS thickness with the PDMS solution volume fraction; it is not fit to bonded thickness.",
            "",
            "## Outputs",
            "",
            "- `comparison_predictions.csv` (bonded thickness)",
            "- `comparison_metrics.csv` (bonded thickness)",
            "- `comparison_plot.png` (bonded thickness)",
            "- `mobile_layer_predictions.csv`",
            "- `mobile_layer_metrics.csv`",
            "- `mobile_layer_comparison_plot.png`",
            "- `comparison_metadata.json`",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Saved comparison results to: {run_dir}")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print("\nMobile-layer metrics:")
    print(mobile_metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
