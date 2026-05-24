"""Analytical thickness models for the PDMS + ethyl acetate comparison workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
DEFAULT_DWELL_TIME_S = 2000.0
DEFAULT_WITHDRAWAL_SPEED_MM_S = 1.0
DEFAULT_FILM_WIDTH_M = 0.065
DEFAULT_DENSITY_KG_M3 = 902.0
DEFAULT_VISCOSITY_CP = 0.423
DEFAULT_SURFACE_TENSION_MN_M = 24.0
DEFAULT_WET_TO_BONDED_RETENTION = 1.0e-3


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


def bonded_layer_adsorption_model(df: pd.DataFrame) -> pd.Series:
    dwell_time_s = _get_dwell_time(df)
    equilibrium_bonded_nm = 0.80
    tau_s = 1000.0
    return (equilibrium_bonded_nm * (1.0 - np.exp(-dwell_time_s / tau_s))).astype(float)


def concentration_dependent_adsorption_time_model(df: pd.DataFrame) -> pd.Series:
    concentration = _get_concentration(df)
    dwell_time_s = _get_dwell_time(df)
    y0_nm = 0.40
    a0_nm = 0.40
    k1_eff = 4.3e-4
    k2 = 2.6e-4
    tau_s = 1.0 / (k1_eff * concentration + k2)
    return (y0_nm + a0_nm * (1.0 - np.exp(-dwell_time_s / tau_s))).astype(float)


def landau_levich_wet_mobile_layer_model(df: pd.DataFrame) -> pd.Series:
    concentration = _get_proxy_concentration(df)
    viscosity_pa_s = _get_viscosity_pa_s(df)
    withdrawal_speed_m_s = _get_withdrawal_speed_m_s(df)
    surface_tension_n_m = _get_surface_tension_n_m(df)
    density_kg_m3 = _get_density_kg_m3(df)
    wet_film_m = 0.94 * (viscosity_pa_s * withdrawal_speed_m_s) ** (2.0 / 3.0)
    wet_film_m /= (surface_tension_n_m ** (1.0 / 6.0)) * np.sqrt(density_kg_m3 * G)
    wet_film_nm = wet_film_m * NM_PER_M
    return (wet_film_nm * concentration * DEFAULT_WET_TO_BONDED_RETENTION).astype(float)


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
