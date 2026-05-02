"""BPNN pipeline for holdout validation with 2000 training and 1000 validation samples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Feature columns
FEATURE_COLUMNS = None
TARGET_COLUMNS = None


def load_holdout_data(split: str = "train") -> pd.DataFrame:
    """Load the holdout validation dataset.
    
    Args:
        split: Either "train" or "validation"
    """
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    
    if split == "train":
        data_path = data_dir / "train_holdout.csv"
    elif split == "validation":
        data_path = data_dir / "validation_holdout.csv"
    else:
        raise ValueError(f"Invalid split: {split}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    return df


def get_feature_target_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Get feature and target columns for the dataset.
    
    Features: Concentration (g/mL), Uncoated Layer (nm), Total Thickness (nm)
    Target: Bonded Thickness (nm)
    """
    global FEATURE_COLUMNS, TARGET_COLUMNS
    
    if FEATURE_COLUMNS is None:
        FEATURE_COLUMNS = ['Concentration (g/mL)', 'Uncoated Layer (nm)', 'Total Thickness (nm)']
    
    if TARGET_COLUMNS is None:
        TARGET_COLUMNS = ['Bonded Thickness (nm)']
    
    return FEATURE_COLUMNS, TARGET_COLUMNS


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into feature and target matrices."""
    feature_cols, target_cols = get_feature_target_columns(df)
    features = df[feature_cols].copy()
    targets = df[target_cols].copy()
    return features, targets


def train_and_evaluate() -> dict:
    """Train BPNN model on holdout training set and evaluate on validation set."""
    # Load data
    train_df = load_holdout_data("train")
    val_df = load_holdout_data("validation")
    
    X_train, y_train = split_features_targets(train_df)
    X_val, y_val = split_features_targets(val_df)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Flatten y values to 1D arrays
    y_train_flat = np.asarray(y_train).flatten()
    y_val_flat = np.asarray(y_val).flatten()
    
    # Train model
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train_flat)
    
    # Evaluate
    y_pred = model.predict(X_val_scaled)
    
    mse = mean_squared_error(y_val_flat, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val_flat, y_pred)
    r2 = r2_score(y_val_flat, y_pred)
    
    metrics = {
        "model": "BPNN",
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "y_pred": y_pred,
        "y_val": y_val_flat,
    }
    
    return metrics


def plot_results(metrics: dict, output_dir: Path | None = None) -> None:
    """Generate and save evaluation plots."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    y_pred = np.asarray(metrics["y_pred"]).flatten()
    y_val = np.asarray(metrics["y_val"]).flatten()
    model_name = metrics["model"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{model_name} - Holdout Validation Results", fontsize=16)
    
    # Predictions vs Actual
    axes[0, 0].scatter(y_val, y_pred, alpha=0.6)
    axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Predictions vs Actual")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_val - y_pred
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
    output_path = output_dir / f"{model_name}_evaluation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    metrics = train_and_evaluate()
    print(f"\n{metrics['model']} Results:")
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  R²:   {metrics['R2']:.6f}")
    
    plot_results(metrics)
