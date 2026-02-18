"""Random Forest Regression (RFR) pipeline with evaluation metrics and visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Feature columns to be determined based on agg.data structure
FEATURE_COLUMNS = None
TARGET_COLUMNS = None


def load_agg_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load the aggregated dataset for experiments."""
    if data_path is None:
        data_path = Path(__file__).resolve().parents[2] / "data" / "agg.data.xlsx"
    
    # Read Excel file
    df = pd.read_excel(data_path)
    return df


def get_feature_target_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Get feature and target columns for the aggregated dataset.
    
    Features: All columns except 'Bonded Thickness (nm)'
    Target: 'Bonded Thickness (nm)'
    """
    global FEATURE_COLUMNS, TARGET_COLUMNS
    
    if FEATURE_COLUMNS is None:
        # Features: everything except bonded thickness
        FEATURE_COLUMNS = ['Concentration (g/mL)', 'Uncoated Layer (nm)', 'Total Thickness (nm)']
    
    if TARGET_COLUMNS is None:
        # Target: bonded thickness
        TARGET_COLUMNS = ['Bonded Thickness (nm)']
    
    return FEATURE_COLUMNS, TARGET_COLUMNS


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into feature and target matrices."""
    feature_cols, target_cols = get_feature_target_columns(df)
    features = df[feature_cols].copy()
    targets = df[target_cols].copy()
    return features, targets


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name: str = "Random Forest Regression") -> dict:
    """Train Random Forest model and return evaluation metrics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Flatten y values to 1D arrays
    y_train_flat = np.asarray(y_train).flatten()
    y_test_flat = np.asarray(y_test).flatten()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
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
    output_dir.mkdir(exist_ok=True)
    
    # Convert to numpy arrays for consistent handling
    import numpy as np
    y_pred = np.asarray(metrics["y_pred"]).flatten()
    y_test = np.asarray(metrics["y_test"]).flatten()
    model_name = metrics["model"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{model_name} - Model Evaluation", fontsize=16)
    
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
    output_path = output_dir / f"{model_name}_evaluation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()
