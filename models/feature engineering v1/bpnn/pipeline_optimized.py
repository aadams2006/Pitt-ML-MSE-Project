"""BPNN pipeline with OPTIMIZED hyperparameters."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Feature columns to be determined based on agg.data structure
FEATURE_COLUMNS = None
TARGET_COLUMNS = None


def load_agg_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load the aggregated dataset for experiments."""
    if data_path is None:
        data_path = Path(__file__).resolve().parents[1] / "data FE-V1" / "synthetic_data_improved.csv"
    
    # Read CSV file
    df = pd.read_csv(data_path)
    return df


def get_feature_target_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Get feature and target columns for the aggregated dataset.
    
    Features: All columns except 'Bonded Thickness (nm)'
    Target: 'Bonded Thickness (nm)'
    """
    global FEATURE_COLUMNS, TARGET_COLUMNS
    
    target_col = "Bonded Thickness (nm)"
    if FEATURE_COLUMNS is None:
        # Features: everything except bonded thickness
        FEATURE_COLUMNS = [col for col in df.columns if col != target_col]
    
    if TARGET_COLUMNS is None:
        # Target: bonded thickness
        TARGET_COLUMNS = [target_col]
    
    return FEATURE_COLUMNS, TARGET_COLUMNS


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into feature and target matrices."""
    feature_cols, target_cols = get_feature_target_columns(df)
    features = df[feature_cols].copy()
    targets = df[target_cols].copy()
    return features, targets


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name: str = "BPNN") -> dict:
    """Train BPNN model with OPTIMIZED hyperparameters and return evaluation metrics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Flatten y values to 1D arrays
    y_train_flat = np.asarray(y_train).flatten()
    y_test_flat = np.asarray(y_test).flatten()
    
    # OPTIMIZED HYPERPARAMETERS:
    # - hidden_layer_sizes: (100, 50, 25) [was (100, 50)]
    # - learning_rate_init: 0.01 [was 0.001]
    # - max_iter: 500 [was 1000]
    # - alpha: 1e-05 [was 0.0001]
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        learning_rate_init=0.01,
        max_iter=500,
        alpha=1e-05,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_scaled, y_train_flat)
    
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test_flat, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred)
    r2 = r2_score(y_test_flat, y_pred)
    
    metrics = {
        "model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "y_pred": y_pred,
        "y_test": y_test_flat,
    }
    
    return metrics


def plot_results(metrics: dict, output_dir: Path | None = None) -> None:
    """Generate and save evaluation plots."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy arrays for consistent handling
    import numpy as np
    y_pred = np.asarray(metrics["y_pred"]).flatten()
    y_test = np.asarray(metrics["y_test"]).flatten()
    model_name = metrics["model"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{model_name} - Model Evaluation (OPTIMIZED)", fontsize=16)
    
    # Predictions vs Actual
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Predictions vs Actual")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residual Plot")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metrics text
    metrics_text = f"""R² Score: {metrics['R2']:.4f}
RMSE: {metrics['RMSE']:.4f}
MAE: {metrics['MAE']:.4f}
MSE: {metrics['MSE']:.4f}"""
    axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=12, family="monospace")
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Performance Metrics")
    
    # Distribution of residuals
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Residual Distribution")
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f"{model_name}_evaluation_optimized.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def save_metrics(metrics: dict, output_dir: Path) -> None:
    """Save evaluation metrics and predictions to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    preds_path = output_dir / "predictions.csv"

    metrics_df = pd.DataFrame([
        {
            "model": metrics["model"],
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
        }
    ])
    metrics_df.to_csv(metrics_path, index=False)

    preds_df = pd.DataFrame({
        "y_true": np.asarray(metrics["y_test"]).flatten(),
        "y_pred": np.asarray(metrics["y_pred"]).flatten(),
    })
    preds_df.to_csv(preds_path, index=False)



def run_experiment(test_size: float = 0.2, random_state: int = 42) -> dict:
    """Run a full train/eval cycle and persist outputs."""
    if "load_agg_data" in globals():
        df = load_agg_data()
    else:
        df = load_synthetic_data()

    X, y = split_features_targets(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model_name = Path(__file__).stem
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, model_name=model_name)

    output_dir = Path(__file__).resolve().parent / "results" / model_name
    save_metrics(metrics, output_dir)
    plot_results(metrics, output_dir=output_dir)

    print(f"Results saved to: {output_dir}")
    return metrics


if __name__ == "__main__":
    run_experiment()
