"""Fit analytical model parameters to experimental data."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTAL_DATA_PATH = BASE_DIR / "ethyl acetate+pdms.csv"
TARGET_COLUMN = "Bonded Thickness (nm)"

FITTED_PARAMS_PATH = BASE_DIR / "artifacts" / "fitted_analytical_params.json"


def load_experimental_data() -> pd.DataFrame:
    df = pd.read_csv(EXPERIMENTAL_DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


def _get_concentration(df: pd.DataFrame) -> pd.Series:
    """Extract concentration with default fallback."""
    if "Concentration (g/mL)" in df.columns:
        return pd.to_numeric(df["Concentration (g/mL)"], errors="coerce").fillna(0.1).clip(lower=0.0)
    return pd.Series(0.1, index=df.index, dtype=float)


def _get_dwell_time(df: pd.DataFrame) -> pd.Series:
    return pd.Series(20.0, index=df.index, dtype=float)


def bonded_layer_adsorption_model_parameterized(df: pd.DataFrame, equilibrium_bonded_nm: float, tau_s: float) -> pd.Series:
    """Bonded layer adsorption with fitted parameters."""
    dwell_time_s = _get_dwell_time(df)
    return (equilibrium_bonded_nm * (1.0 - np.exp(-dwell_time_s / tau_s))).astype(float)


def concentration_dependent_adsorption_model_parameterized(
    df: pd.DataFrame, y0_nm: float, a0_nm: float, k1_eff: float, k2: float
) -> pd.Series:
    """Concentration-dependent adsorption with fitted parameters."""
    concentration = _get_concentration(df)
    dwell_time_s = _get_dwell_time(df)
    tau_s = 1.0 / (k1_eff * concentration + k2)
    return (y0_nm + a0_nm * (1.0 - np.exp(-dwell_time_s / tau_s))).astype(float)


def fit_bonded_layer_adsorption(y_true: np.ndarray, df: pd.DataFrame) -> dict:
    """Fit bonded layer adsorption model parameters."""
    def objective(params):
        equilibrium_bonded_nm, tau_s = params
        # Ensure reasonable bounds
        if equilibrium_bonded_nm <= 0 or tau_s <= 0:
            return 1e10
        y_pred = bonded_layer_adsorption_model_parameterized(df, equilibrium_bonded_nm, tau_s).values
        mse = mean_squared_error(y_true, y_pred)
        return mse

    # Use differential evolution for global optimization
    bounds = [(0.1, 2.0), (0.5, 100.0)]  # equilibrium, tau
    result = differential_evolution(objective, bounds, seed=42, maxiter=300)
    
    equilibrium_bonded_nm, tau_s = result.x
    y_pred = bonded_layer_adsorption_model_parameterized(df, equilibrium_bonded_nm, tau_s).values
    
    return {
        "model": "Bonded-Layer Adsorption",
        "equilibrium_bonded_nm": float(equilibrium_bonded_nm),
        "tau_s": float(tau_s),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_concentration_dependent_adsorption(y_true: np.ndarray, df: pd.DataFrame) -> dict:
    """Fit concentration-dependent adsorption model parameters."""
    def objective(params):
        y0_nm, a0_nm, k1_eff, k2 = params
        # Ensure reasonable bounds and positive values
        if any(p <= 0 for p in params):
            return 1e10
        y_pred = concentration_dependent_adsorption_model_parameterized(df, y0_nm, a0_nm, k1_eff, k2).values
        mse = mean_squared_error(y_true, y_pred)
        return mse

    # Use differential evolution for global optimization
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.01), (0.0, 0.01)]  # y0, a0, k1_eff, k2
    result = differential_evolution(objective, bounds, seed=42, maxiter=500)
    
    y0_nm, a0_nm, k1_eff, k2 = result.x
    y_pred = concentration_dependent_adsorption_model_parameterized(df, y0_nm, a0_nm, k1_eff, k2).values
    
    return {
        "model": "Concentration-Dependent Adsorption Time",
        "y0_nm": float(y0_nm),
        "a0_nm": float(a0_nm),
        "k1_eff": float(k1_eff),
        "k2": float(k2),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    df = load_experimental_data()
    y_true = df[TARGET_COLUMN].values
    
    print("Fitting analytical model parameters to experimental data...")
    print(f"Experimental data: {len(df)} samples")
    print(f"Target: {TARGET_COLUMN}, range: [{y_true.min():.3f}, {y_true.max():.3f}] nm")
    print()
    
    # Fit each model
    fitted_params = {}
    
    print("1. Fitting Bonded-Layer Adsorption model...")
    params1 = fit_bonded_layer_adsorption(y_true, df)
    print(f"   Parameters: equilibrium_bonded_nm={params1['equilibrium_bonded_nm']:.4f}, tau_s={params1['tau_s']:.4f}")
    print(f"   R2={params1['r2']:.4f}, RMSE={params1['rmse']:.4f}")
    fitted_params["bonded_layer_adsorption"] = params1
    print()
    
    print("2. Fitting Concentration-Dependent Adsorption model...")
    params2 = fit_concentration_dependent_adsorption(y_true, df)
    print(f"   Parameters: y0_nm={params2['y0_nm']:.4f}, a0_nm={params2['a0_nm']:.4f}, k1_eff={params2['k1_eff']:.6f}, k2={params2['k2']:.6f}")
    print(f"   R2={params2['r2']:.4f}, RMSE={params2['rmse']:.4f}")
    fitted_params["concentration_dependent_adsorption"] = params2
    print()
    
    # Save fitted parameters
    fitted_params["timestamp"] = datetime.now().isoformat()
    fitted_params["experimental_data"] = str(EXPERIMENTAL_DATA_PATH)
    fitted_params["n_samples"] = len(df)
    
    FITTED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FITTED_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(fitted_params, f, indent=2)
    
    print(f"Fitted parameters saved to: {FITTED_PARAMS_PATH}")


if __name__ == "__main__":
    main()
