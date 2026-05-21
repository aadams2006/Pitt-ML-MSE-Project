"""Run side-by-side evaluation of the best ML model and analytical models."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from analytical_models import (
    DEFAULT_DENSITY_KG_M3,
    DEFAULT_DWELL_TIME_S,
    DEFAULT_EVAPORATION_RATE_M_S,
    DEFAULT_FILM_WIDTH_M,
    DEFAULT_WITHDRAWAL_SPEED_MM_S,
    EXPERIMENT_SOLUTE,
    EXPERIMENT_SOLVENT,
    get_analytical_models,
)


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
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

    analytical_models = get_analytical_models()
    for analytical_model in analytical_models:
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

    comparison_metadata = {
        "created_at": datetime.now().isoformat(),
        "experimental_data_path": str(EXPERIMENTAL_DATA_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "best_model_training_summary": str(BEST_MODEL_METADATA_PATH),
        "target_column": TARGET_COLUMN,
        "feature_order": feature_order,
        "hexane_properties_used_for_ml_only": HEXANE_PROPERTIES,
        "analytical_model_experiment_identity": {
            "solute": EXPERIMENT_SOLUTE,
            "solvent": EXPERIMENT_SOLVENT,
        },
        "fixed_experimental_constants_used_by_analytical_models": {
            "dwell_time_s": DEFAULT_DWELL_TIME_S,
            "withdrawal_speed_mm_s": DEFAULT_WITHDRAWAL_SPEED_MM_S,
            "film_width_m": DEFAULT_FILM_WIDTH_M,
            "evaporation_rate_m_s": DEFAULT_EVAPORATION_RATE_M_S,
            "density_kg_m3": DEFAULT_DENSITY_KG_M3,
        },
        "analytical_models_registered": [model.name for model in analytical_models],
        "n_rows_evaluated": int(len(experimental_df)),
    }
    with open(run_dir / "comparison_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(comparison_metadata, handle, indent=2)

    save_scatter_plot(predictions_df, run_dir / "comparison_plot.png")

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
        "- Dwell time, withdrawal speed, film width, evaporation rate, and density are fixed experiment-level constants in the current implementation.",
        "- The current evaporation-rate value is a temporary placeholder and not yet tied to a cited source.",
        "- The density used in the Landau-Levich term is the coating-solution density, currently approximated by hexane for the dilute PDMS + hexane bath.",
        "",
        "## Models Included",
        "",
        "- `Bayesian Optimized RF`",
    ]
    if analytical_models:
        for analytical_model in analytical_models:
            readme_lines.append(f"- `{analytical_model.name}`")
    else:
        readme_lines.append("- No analytical formulas registered yet.")
    readme_lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `comparison_predictions.csv`",
            "- `comparison_metrics.csv`",
            "- `comparison_metadata.json`",
            "- `comparison_plot.png`",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Saved comparison results to: {run_dir}")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
