"""Analytical thickness models for the PDMS + ethyl acetate comparison workflow.

The analytical predictors are constrained to preserve physically reasonable
trend directions so they do not collapse into near-constant bonded-thickness
outputs:
- higher concentration should not reduce bonded thickness,
- higher total thickness should not reduce bonded thickness,
- higher uncoated layer should not increase bonded thickness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import json
from pathlib import Path

import numpy as np
import pandas as pd


G = 9.81
NM_PER_M = 1e9

EXPERIMENT_SOLUTE = "PDMS"
EXPERIMENT_SOLVENT = "ethyl acetate"
RELATIVE_EVAPORATION_REFERENCE = {
    "source": None,
    "relative_evaporation_buac_equals_1": None,
    "note": "No solvent-specific relative evaporation reference is currently recorded for ethyl acetate. E remains symbolic.",
}
DEFAULT_DWELL_TIME_S = 20.0
DEFAULT_WITHDRAWAL_SPEED_MM_S = 1.0
DEFAULT_FILM_WIDTH_M = 0.065
DEFAULT_DENSITY_KG_M3 = 902.0
DEFAULT_VISCOSITY_CP = 0.423
DEFAULT_SURFACE_TENSION_MN_M = 24.0
DEFAULT_WET_TO_BONDED_RETENTION = 1.0e-3
MIN_POSITIVE = 1.0e-9


def _load_fitted_parameters() -> dict:
    """Load fitted analytical model parameters from JSON file if available."""
    fitted_params_path = Path(__file__).resolve().parents[1] / "artifacts" / "fitted_analytical_params.json"
    if fitted_params_path.exists():
        with open(fitted_params_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Load fitted parameters at module initialization
_FITTED_PARAMS = _load_fitted_parameters()


@dataclass(frozen=True)
class AnalyticalModel:
    name: str
    description: str
    predictor: Callable[[pd.DataFrame], pd.Series]
    requires_effective_e: bool = False
    symbolic_expression: str | None = None

    def predict(self, df: pd.DataFrame) -> pd.Series:
        prediction = self.predictor(df).copy()
        prediction.index = df.index
        return prediction.rename(self.name)


def _series_or_default(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _get_concentration(df: pd.DataFrame) -> pd.Series:
    return _series_or_default(df, "Concentration (g/mL)", 0.1).clip(lower=0.0)


def _get_proxy_concentration(df: pd.DataFrame) -> pd.Series:
    concentration = _get_concentration(df)
    scale = max(float(concentration.max()), 1.0)
    return concentration / scale


def _get_total_thickness(df: pd.DataFrame) -> pd.Series:
    return _series_or_default(df, "Total Thickness (nm)", 1.0).clip(lower=MIN_POSITIVE)


def _get_uncoated_layer(df: pd.DataFrame) -> pd.Series:
    return _series_or_default(df, "Uncoated Layer (nm)", 1.0).clip(lower=MIN_POSITIVE)


def _get_dwell_time(df: pd.DataFrame) -> pd.Series:
    return pd.Series(DEFAULT_DWELL_TIME_S, index=df.index, dtype=float)


def _get_withdrawal_speed_m_s(df: pd.DataFrame) -> pd.Series:
    return pd.Series(DEFAULT_WITHDRAWAL_SPEED_MM_S / 1000.0, index=df.index, dtype=float)


def _get_viscosity_pa_s(df: pd.DataFrame) -> pd.Series:
    viscosity_cp = _series_or_default(df, "Viscosity (cP)", DEFAULT_VISCOSITY_CP).clip(lower=1e-9)
    return viscosity_cp * 1.0e-3


def _get_surface_tension_n_m(df: pd.DataFrame) -> pd.Series:
    surface_tension_mn_m = _series_or_default(
        df,
        "Surface Tension (mN/m)",
        DEFAULT_SURFACE_TENSION_MN_M,
    ).clip(lower=1e-9)
    return surface_tension_mn_m * 1.0e-3


def _get_density_kg_m3(df: pd.DataFrame) -> pd.Series:
    return pd.Series(DEFAULT_DENSITY_KG_M3, index=df.index, dtype=float)


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


def _get_model_parameters(model_key: str, defaults: dict[str, float]) -> dict[str, float]:
    fitted = _FITTED_PARAMS.get(model_key, {})
    return {
        key: float(fitted.get(key, default_value))
        for key, default_value in defaults.items()
    }


def bonded_layer_adsorption_model(df: pd.DataFrame) -> pd.Series:
    dwell_time_s = _get_dwell_time(df)
    params = _get_model_parameters(
        "bonded_layer_adsorption",
        {
            "equilibrium_scale_nm": 0.80,
            "tau_s": 20.0,
            "concentration_exponent": 0.50,
            "total_thickness_exponent": 0.50,
            "uncoated_layer_exponent": 1.00,
        },
    )
    driver = _deposition_driver(
        df,
        params["concentration_exponent"],
        params["total_thickness_exponent"],
        params["uncoated_layer_exponent"],
    )
    prediction = params["equilibrium_scale_nm"] * driver
    prediction *= 1.0 - np.exp(-(dwell_time_s * driver) / max(params["tau_s"], MIN_POSITIVE))
    return prediction.clip(lower=0.0).astype(float)


def concentration_dependent_adsorption_time_model(df: pd.DataFrame) -> pd.Series:
    dwell_time_s = _get_dwell_time(df)
    params = _get_model_parameters(
        "concentration_dependent_adsorption",
        {
            "y0_nm": 0.20,
            "a0_nm": 0.60,
            "k1_eff": 0.05,
            "k2": 0.01,
            "concentration_exponent": 0.75,
            "total_thickness_exponent": 0.50,
            "uncoated_layer_exponent": 1.00,
        },
    )
    driver = _deposition_driver(
        df,
        params["concentration_exponent"],
        params["total_thickness_exponent"],
        params["uncoated_layer_exponent"],
    )
    tau_s = 1.0 / (
        max(params["k1_eff"], 0.0) * driver + max(params["k2"], MIN_POSITIVE)
    )
    prediction = params["y0_nm"] + params["a0_nm"] * (1.0 - np.exp(-dwell_time_s / tau_s))
    return prediction.clip(lower=0.0).astype(float)


def landau_levich_wet_mobile_layer_model(df: pd.DataFrame) -> pd.Series:
    viscosity_pa_s = _get_viscosity_pa_s(df)
    withdrawal_speed_m_s = _get_withdrawal_speed_m_s(df)
    surface_tension_n_m = _get_surface_tension_n_m(df)
    density_kg_m3 = _get_density_kg_m3(df)
    params = _get_model_parameters(
        "landau_levich_wet_mobile_layer",
        {
            "retention_scale": DEFAULT_WET_TO_BONDED_RETENTION,
            "concentration_exponent": 0.75,
            "total_thickness_exponent": 0.50,
            "uncoated_layer_exponent": 1.00,
        },
    )
    wet_film_m = 0.94 * (viscosity_pa_s * withdrawal_speed_m_s) ** (2.0 / 3.0)
    wet_film_m /= (surface_tension_n_m ** (1.0 / 6.0)) * np.sqrt(density_kg_m3 * G)
    wet_film_nm = wet_film_m * NM_PER_M
    driver = _deposition_driver(
        df,
        params["concentration_exponent"],
        params["total_thickness_exponent"],
        params["uncoated_layer_exponent"],
    )
    prediction = wet_film_nm * max(params["retention_scale"], MIN_POSITIVE) * driver
    return prediction.clip(lower=0.0).astype(float)


def capillarity_evaporation_regime_model(df: pd.DataFrame) -> pd.Series:
    raise RuntimeError("This model requires an effective evaporation rate E and is intentionally left symbolic.")


def combined_capillarity_landau_levich_model(df: pd.DataFrame) -> pd.Series:
    raise RuntimeError("This model requires an effective evaporation rate E and is intentionally left symbolic.")


def get_analytical_models() -> list[AnalyticalModel]:
    return [
        AnalyticalModel(
            name="Bonded-Layer Adsorption",
            description="Exponential bonded-layer growth with a fixed characteristic adsorption time.",
            predictor=bonded_layer_adsorption_model,
        ),
        AnalyticalModel(
            name="Concentration-Dependent Adsorption Time",
            description="Pseudo-first-order adsorption model with concentration-dependent time constant.",
            predictor=concentration_dependent_adsorption_time_model,
        ),
        AnalyticalModel(
            name="Landau-Levich Wet/Mobile Layer",
            description="Classical wet-film entrainment converted to a bonded-thickness proxy.",
            predictor=landau_levich_wet_mobile_layer_model,
        ),
        AnalyticalModel(
            name="Capillarity / Evaporation Regime",
            description="Capillary-feed / evaporation-controlled model, kept symbolic in terms of E.",
            predictor=capillarity_evaporation_regime_model,
            requires_effective_e=True,
            symbolic_expression="h_cap = k_i E / (L U)",
        ),
        AnalyticalModel(
            name="Combined Capillarity + Landau-Levich",
            description="Superposed capillarity and draining model, kept symbolic in terms of E.",
            predictor=combined_capillarity_landau_levich_model,
            requires_effective_e=True,
            symbolic_expression="h_f = k_i E / (L U) + D U^(2/3)",
        ),
    ]
