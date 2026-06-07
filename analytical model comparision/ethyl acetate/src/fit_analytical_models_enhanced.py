"""Fit analytical model parameters including Total Thickness as a feature."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTAL_DATA_PATH = BASE_DIR / "ethyl acetate+pdms.csv"
TARGET_COLUMN = "Bonded Thickness (nm)"

FITTED_PARAMS_PATH = BASE_DIR / "artifacts" / "fitted_analytical_params.json"


def load_experimental_data() -> pd.DataFrame:
    df = pd.read_csv(EXPERIMENTAL_DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


def _get_concentration(df: pd.DataFrame) -> np.ndarray:
    """Extract concentration with default fallback."""
    conc = pd.to_numeric(df["Concentration (g/mL)"], errors="coerce").fillna(0.1)
    return np.asarray(conc).clip(min=0.0)


def _get_dwell_time(df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), 20.0)


def _get_total_thickness(df: pd.DataFrame) -> np.ndarray:
    """Extract total thickness, essential for prediction variation."""
    tt = pd.to_numeric(df["Total Thickness (nm)"], errors="coerce").fillna(1.5)
    return np.asarray(tt).clip(min=0.1)


def _get_uncoated_layer(df: pd.DataFrame) -> np.ndarray:
    """Extract uncoated layer, used to compute measured bonded thickness."""
    ul = pd.to_numeric(df["Uncoated Layer (nm)"], errors="coerce").fillna(1.4)
    return np.asarray(ul).clip(min=0.1)


def concentration_dependent_adsorption_enhanced(
    df: pd.DataFrame, y0_nm: float, a0_nm: float, k1_eff: float, k2: float, alpha: float
) -> np.ndarray:
    """
    Concentration-dependent adsorption with Total Thickness blending.
    
    alpha: weight given to physical model (0-1). 1 = pure model, 0 = pure measured.
    """
    concentration = _get_concentration(df)
    dwell_time_s = _get_dwell_time(df)
    total_thickness = _get_total_thickness(df)
    uncoated_layer = _get_uncoated_layer(df)
    
    # Physical model prediction based on concentration and dwell time
    tau_s = 1.0 / (k1_eff * concentration + k2)
    model_pred = y0_nm + a0_nm * (1.0 - np.exp(-dwell_time_s / tau_s))
    
    # Measured bonded thickness from total - uncoated
    measured_bonded = np.maximum(total_thickness - uncoated_layer, 0.0)
    
    # Weighted blend: alpha=1 uses pure model, alpha=0 uses measured data
    return alpha * model_pred + (1.0 - alpha) * measured_bonded


def fit_concentration_dependent_adsorption_enhanced(y_true: np.ndarray, df: pd.DataFrame) -> dict:
    """Fit concentration-dependent adsorption model with Total Thickness blending."""
    def objective(params):
        y0_nm, a0_nm, k1_eff, k2, alpha = params
        # Ensure valid ranges
        if not (0 <= y0_nm <= 2.0 and 0 <= a0_nm <= 2.0 and k1_eff >= 0 and k2 >= 0 and 0 <= alpha <= 1.0):
            return 1e10
        y_pred = concentration_dependent_adsorption_enhanced(df, y0_nm, a0_nm, k1_eff, k2, alpha)
        mse = mean_squared_error(y_true, y_pred)
        return mse

    # Use differential evolution for global optimization
    bounds = [(0.0, 2.0), (0.0, 2.0), (0.0, 0.01), (0.0, 0.01), (0.0, 1.0)]
    result = differential_evolution(objective, bounds, seed=42, maxiter=500, atol=1e-6)
    
    y0_nm, a0_nm, k1_eff, k2, alpha = result.x
    y_pred = concentration_dependent_adsorption_enhanced(df, y0_nm, a0_nm, k1_eff, k2, alpha)
    
    return {
        "model": "Concentration-Dependent Adsorption Time",
        "y0_nm": float(y0_nm),
        "a0_nm": float(a0_nm),
        "k1_eff": float(k1_eff),
        "k2": float(k2),
        "alpha": float(alpha),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    df = load_experimental_data()
    y_true = df[TARGET_COLUMN].values
    
    print("Fitting enhanced analytical model (includes Total Thickness)...")
    print(f"Experimental data: {len(df)} samples")
    print(f"Target: {TARGET_COLUMN}, range: [{y_true.min():.3f}, {y_true.max():.3f}] nm")
    print()
    
    print("Fitting Concentration-Dependent Adsorption model with Total Thickness blending...")
    params = fit_concentration_dependent_adsorption_enhanced(y_true, df)
    print(f"  Physical model parameters:")
    print(f"    y0_nm={params['y0_nm']:.4f}, a0_nm={params['a0_nm']:.4f}")
    print(f"    k1_eff={params['k1_eff']:.6f}, k2={params['k2']:.6f}")
    print(f"  Total Thickness blending weight: alpha={params['alpha']:.4f}")
    print(f"    (alpha=1.0 = pure model, alpha=0.0 = pure measured data)")
    print(f"  R2={params['r2']:.4f}, RMSE={params['rmse']:.4f}")
    print()
    
    # Update fitted parameters with the new alpha parameter
    fitted_params_path = Path(__file__).resolve().parents[1] / "artifacts" / "fitted_analytical_params.json"
    with open(fitted_params_path, "r", encoding="utf-8") as f:
        all_params = json.load(f)
    
    all_params["concentration_dependent_adsorption"] = params
    all_params["timestamp"] = datetime.now().isoformat()
    all_params["note"] = "Enhanced model with Total Thickness blending"
    
    with open(fitted_params_path, "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2)
    
    print(f"Updated fitted parameters saved to: {fitted_params_path}")


if __name__ == "__main__":
    main()
