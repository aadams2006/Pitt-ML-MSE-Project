"""Fit analytical model parameters to experimental data for toluene."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTAL_DATA_PATH = BASE_DIR / "toluene+pdms.csv"
TARGET_COLUMN = "Bonded Thickness (nm)"

FITTED_PARAMS_PATH = BASE_DIR / "artifacts" / "fitted_analytical_params.json"
DEFAULT_DWELL_TIME_S = 20.0
DEFAULT_DENSITY_KG_M3 = 867.0
DEFAULT_VISCOSITY_CP = 0.68
DEFAULT_SURFACE_TENSION_MN_M = 29.46
DEFAULT_WITHDRAWAL_SPEED_MM_S = 1.0
G = 9.81
NM_PER_M = 1e9
MIN_POSITIVE = 1.0e-9


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
    return pd.Series(DEFAULT_DWELL_TIME_S, index=df.index, dtype=float)


def _get_total_thickness(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["Total Thickness (nm)"], errors="coerce").fillna(1.0).clip(lower=MIN_POSITIVE)


def _get_uncoated_layer(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["Uncoated Layer (nm)"], errors="coerce").fillna(1.0).clip(lower=MIN_POSITIVE)


def _median_safe(series: pd.Series) -> float:
    median_value = float(series.median())
    return max(median_value, MIN_POSITIVE)


def _deposition_driver(
    df: pd.DataFrame,
    concentration_exponent: float,
    total_thickness_exponent: float,
    uncoated_layer_exponent: float,
) -> pd.Series:
    concentration = _get_concentration(df).clip(lower=MIN_POSITIVE)
    total_thickness = _get_total_thickness(df)
    uncoated_layer = _get_uncoated_layer(df)

    concentration_ref = _median_safe(concentration)
    total_thickness_ref = _median_safe(total_thickness)
    uncoated_layer_ref = _median_safe(uncoated_layer)

    concentration_term = (concentration / concentration_ref) ** concentration_exponent
    total_thickness_term = (total_thickness / total_thickness_ref) ** total_thickness_exponent
    uncoated_layer_term = (uncoated_layer_ref / uncoated_layer) ** uncoated_layer_exponent
    return (concentration_term * total_thickness_term * uncoated_layer_term).astype(float)


def _wet_film_nm(df: pd.DataFrame) -> pd.Series:
    viscosity_pa_s = pd.Series(DEFAULT_VISCOSITY_CP * 1.0e-3, index=df.index, dtype=float)
    withdrawal_speed_m_s = pd.Series(DEFAULT_WITHDRAWAL_SPEED_MM_S / 1000.0, index=df.index, dtype=float)
    surface_tension_n_m = pd.Series(DEFAULT_SURFACE_TENSION_MN_M * 1.0e-3, index=df.index, dtype=float)
    density_kg_m3 = pd.Series(DEFAULT_DENSITY_KG_M3, index=df.index, dtype=float)

    wet_film_m = 0.94 * (viscosity_pa_s * withdrawal_speed_m_s) ** (2.0 / 3.0)
    wet_film_m /= (surface_tension_n_m ** (1.0 / 6.0)) * np.sqrt(density_kg_m3 * G)
    return wet_film_m * NM_PER_M


def bonded_layer_adsorption_model_parameterized(
    df: pd.DataFrame,
    equilibrium_scale_nm: float,
    tau_s: float,
    concentration_exponent: float,
    total_thickness_exponent: float,
    uncoated_layer_exponent: float,
) -> pd.Series:
    """Bonded layer adsorption with fitted trend-aware parameters."""
    dwell_time_s = _get_dwell_time(df)
    driver = _deposition_driver(
        df,
        concentration_exponent,
        total_thickness_exponent,
        uncoated_layer_exponent,
    )
    prediction = equilibrium_scale_nm * driver
    prediction *= 1.0 - np.exp(-(dwell_time_s * driver) / max(tau_s, MIN_POSITIVE))
    return prediction.clip(lower=0.0).astype(float)


def concentration_dependent_adsorption_model_parameterized(
    df: pd.DataFrame,
    y0_nm: float,
    a0_nm: float,
    k1_eff: float,
    k2: float,
    concentration_exponent: float,
    total_thickness_exponent: float,
    uncoated_layer_exponent: float,
) -> pd.Series:
    """Concentration-dependent adsorption with fitted trend-aware parameters."""
    dwell_time_s = _get_dwell_time(df)
    driver = _deposition_driver(
        df,
        concentration_exponent,
        total_thickness_exponent,
        uncoated_layer_exponent,
    )
    tau_s = 1.0 / (max(k1_eff, 0.0) * driver + max(k2, MIN_POSITIVE))
    prediction = y0_nm + a0_nm * (1.0 - np.exp(-dwell_time_s / tau_s))
    return prediction.clip(lower=0.0).astype(float)


def landau_levich_wet_mobile_layer_model_parameterized(
    df: pd.DataFrame,
    retention_scale: float,
    concentration_exponent: float,
    total_thickness_exponent: float,
    uncoated_layer_exponent: float,
) -> pd.Series:
    """Landau-Levich wet-film retention proxy with fitted trend-aware parameters."""
    driver = _deposition_driver(
        df,
        concentration_exponent,
        total_thickness_exponent,
        uncoated_layer_exponent,
    )
    prediction = _wet_film_nm(df) * max(retention_scale, MIN_POSITIVE) * driver
    return prediction.clip(lower=0.0).astype(float)


def fit_bonded_layer_adsorption(y_true: np.ndarray, df: pd.DataFrame) -> dict:
    """Fit bonded layer adsorption model parameters."""
    def objective(params):
        equilibrium_scale_nm, tau_s, concentration_exponent, total_thickness_exponent, uncoated_layer_exponent = params
        if equilibrium_scale_nm <= 0 or tau_s <= 0:
            return 1e10
        y_pred = bonded_layer_adsorption_model_parameterized(
            df,
            equilibrium_scale_nm,
            tau_s,
            concentration_exponent,
            total_thickness_exponent,
            uncoated_layer_exponent,
        ).values
        return mean_squared_error(y_true, y_pred)

    bounds = [
        (0.01, 2.0),
        (0.1, 200.0),
        (0.0, 3.0),
        (0.0, 3.0),
        (0.0, 4.0),
    ]
    result = differential_evolution(objective, bounds, seed=42, maxiter=120, polish=True)

    equilibrium_scale_nm, tau_s, concentration_exponent, total_thickness_exponent, uncoated_layer_exponent = result.x
    y_pred = bonded_layer_adsorption_model_parameterized(
        df,
        equilibrium_scale_nm,
        tau_s,
        concentration_exponent,
        total_thickness_exponent,
        uncoated_layer_exponent,
    ).values

    return {
        "model": "Bonded-Layer Adsorption",
        "equilibrium_scale_nm": float(equilibrium_scale_nm),
        "tau_s": float(tau_s),
        "concentration_exponent": float(concentration_exponent),
        "total_thickness_exponent": float(total_thickness_exponent),
        "uncoated_layer_exponent": float(uncoated_layer_exponent),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "target_correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def fit_concentration_dependent_adsorption(y_true: np.ndarray, df: pd.DataFrame) -> dict:
    """Fit concentration-dependent adsorption model parameters."""
    def objective(params):
        y0_nm, a0_nm, k1_eff, k2, concentration_exponent, total_thickness_exponent, uncoated_layer_exponent = params
        if y0_nm < 0 or a0_nm < 0 or k1_eff < 0 or k2 < 0:
            return 1e10
        y_pred = concentration_dependent_adsorption_model_parameterized(
            df,
            y0_nm,
            a0_nm,
            k1_eff,
            k2,
            concentration_exponent,
            total_thickness_exponent,
            uncoated_layer_exponent,
        ).values
        return mean_squared_error(y_true, y_pred)

    bounds = [
        (0.0, 1.0),
        (0.0, 2.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 3.0),
        (0.0, 3.0),
        (0.0, 4.0),
    ]
    result = differential_evolution(objective, bounds, seed=42, maxiter=160, polish=True)

    y0_nm, a0_nm, k1_eff, k2, concentration_exponent, total_thickness_exponent, uncoated_layer_exponent = result.x
    y_pred = concentration_dependent_adsorption_model_parameterized(
        df,
        y0_nm,
        a0_nm,
        k1_eff,
        k2,
        concentration_exponent,
        total_thickness_exponent,
        uncoated_layer_exponent,
    ).values

    return {
        "model": "Concentration-Dependent Adsorption Time",
        "y0_nm": float(y0_nm),
        "a0_nm": float(a0_nm),
        "k1_eff": float(k1_eff),
        "k2": float(k2),
        "concentration_exponent": float(concentration_exponent),
        "total_thickness_exponent": float(total_thickness_exponent),
        "uncoated_layer_exponent": float(uncoated_layer_exponent),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "target_correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def fit_landau_levich_wet_mobile_layer(y_true: np.ndarray, df: pd.DataFrame) -> dict:
    """Fit the bonded-retention proxy for the Landau-Levich model."""
    def objective(params):
        retention_scale, concentration_exponent, total_thickness_exponent, uncoated_layer_exponent = params
        if retention_scale <= 0:
            return 1e10
        y_pred = landau_levich_wet_mobile_layer_model_parameterized(
            df,
            retention_scale,
            concentration_exponent,
            total_thickness_exponent,
            uncoated_layer_exponent,
        ).values
        return mean_squared_error(y_true, y_pred)

    bounds = [
        (1.0e-6, 1.0e-2),
        (0.0, 3.0),
        (0.0, 3.0),
        (0.0, 4.0),
    ]
    result = differential_evolution(objective, bounds, seed=42, maxiter=120, polish=True)

    retention_scale, concentration_exponent, total_thickness_exponent, uncoated_layer_exponent = result.x
    y_pred = landau_levich_wet_mobile_layer_model_parameterized(
        df,
        retention_scale,
        concentration_exponent,
        total_thickness_exponent,
        uncoated_layer_exponent,
    ).values

    return {
        "model": "Landau-Levich Wet/Mobile Layer",
        "retention_scale": float(retention_scale),
        "concentration_exponent": float(concentration_exponent),
        "total_thickness_exponent": float(total_thickness_exponent),
        "uncoated_layer_exponent": float(uncoated_layer_exponent),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "target_correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
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
    print(
        "   Parameters: "
        f"equilibrium_scale_nm={params1['equilibrium_scale_nm']:.4f}, "
        f"tau_s={params1['tau_s']:.4f}, "
        f"conc_exp={params1['concentration_exponent']:.4f}, "
        f"total_exp={params1['total_thickness_exponent']:.4f}, "
        f"uncoated_exp={params1['uncoated_layer_exponent']:.4f}"
    )
    print(
        f"   R2={params1['r2']:.4f}, RMSE={params1['rmse']:.4f}, "
        f"Corr={params1['target_correlation']:.4f}, PredStd={params1['prediction_std']:.4f}"
    )
    fitted_params["bonded_layer_adsorption"] = params1
    print()
    
    print("2. Fitting Concentration-Dependent Adsorption model...")
    params2 = fit_concentration_dependent_adsorption(y_true, df)
    print(
        "   Parameters: "
        f"y0_nm={params2['y0_nm']:.4f}, "
        f"a0_nm={params2['a0_nm']:.4f}, "
        f"k1_eff={params2['k1_eff']:.6f}, "
        f"k2={params2['k2']:.6f}, "
        f"conc_exp={params2['concentration_exponent']:.4f}, "
        f"total_exp={params2['total_thickness_exponent']:.4f}, "
        f"uncoated_exp={params2['uncoated_layer_exponent']:.4f}"
    )
    print(
        f"   R2={params2['r2']:.4f}, RMSE={params2['rmse']:.4f}, "
        f"Corr={params2['target_correlation']:.4f}, PredStd={params2['prediction_std']:.4f}"
    )
    fitted_params["concentration_dependent_adsorption"] = params2
    print()

    print("3. Fitting Landau-Levich Wet/Mobile Layer model...")
    params3 = fit_landau_levich_wet_mobile_layer(y_true, df)
    print(
        "   Parameters: "
        f"retention_scale={params3['retention_scale']:.6f}, "
        f"conc_exp={params3['concentration_exponent']:.4f}, "
        f"total_exp={params3['total_thickness_exponent']:.4f}, "
        f"uncoated_exp={params3['uncoated_layer_exponent']:.4f}"
    )
    print(
        f"   R2={params3['r2']:.4f}, RMSE={params3['rmse']:.4f}, "
        f"Corr={params3['target_correlation']:.4f}, PredStd={params3['prediction_std']:.4f}"
    )
    fitted_params["landau_levich_wet_mobile_layer"] = params3
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
