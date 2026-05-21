"""Analytical thickness models used by the comparison runner.

The experimental PDMS dataset does not contain every process variable required by
the literature formulas (for example withdrawal speed and dwell time). To keep the
models executable on the available table, each model uses documented fallback
assumptions when a required column is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


G = 9.81
NM_PER_M = 1e9

# Fallback assumptions used when the experimental dataframe does not contain
# the process variables required by the literature expressions.
DEFAULT_DWELL_TIME_S = 2000.0
DEFAULT_WITHDRAWAL_SPEED_MM_S = 1.0
DEFAULT_FILM_WIDTH_M = 0.065
DEFAULT_EVAPORATION_RATE_M_S = 1.0e-7
DEFAULT_DENSITY_KG_M3 = 655.0
DEFAULT_WET_TO_BONDED_RETENTION = 1.0e-3


@dataclass(frozen=True)
class AnalyticalModel:
    """Minimal analytical-model interface for the comparison runner."""

    name: str
    description: str
    predictor: Callable[[pd.DataFrame], pd.Series]

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return model predictions aligned to the input dataframe index."""
        prediction = self.predictor(df).copy()
        prediction.index = df.index
        return prediction.rename(self.name)


def _series_or_default(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    """Return a numeric series or a constant fallback series."""
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _get_concentration(df: pd.DataFrame) -> pd.Series:
    """Return concentration using the project column name."""
    return _series_or_default(df, "Concentration (g/mL)", 0.1).clip(lower=0.0)


def _get_proxy_concentration(df: pd.DataFrame) -> pd.Series:
    """
    Return a normalized concentration term for proxy hydrodynamic models.

    The project dataset concentration scale is much larger than the concentration
    windows used in several literature studies. For executable proxy formulas, the
    concentration term is normalized to the dataset maximum so the analytical
    expressions remain row-varying without blowing up numerically.
    """
    concentration = _get_concentration(df)
    scale = max(float(concentration.max()), 1.0)
    return concentration / scale


def _get_dwell_time(df: pd.DataFrame) -> pd.Series:
    """Return dwell time in seconds, using fallback literature conditions."""
    return _series_or_default(df, "Dwell Time (s)", DEFAULT_DWELL_TIME_S).clip(lower=0.0)


def _get_withdrawal_speed_m_s(df: pd.DataFrame) -> pd.Series:
    """Return withdrawal speed in SI units."""
    speed_mm_s = _series_or_default(
        df,
        "Withdrawal Speed (mm/s)",
        DEFAULT_WITHDRAWAL_SPEED_MM_S,
    ).clip(lower=1e-9)
    return speed_mm_s / 1000.0


def _get_viscosity_pa_s(df: pd.DataFrame) -> pd.Series:
    """Return viscosity in Pa*s, converting from cP when present."""
    viscosity_cp = _series_or_default(df, "Viscosity (cP)", 0.377).clip(lower=1e-9)
    return viscosity_cp * 1.0e-3


def _get_surface_tension_n_m(df: pd.DataFrame) -> pd.Series:
    """Return surface tension in N/m, converting from mN/m when present."""
    surface_tension_mn_m = _series_or_default(df, "Surface Tension (mN/m)", 17.89).clip(lower=1e-9)
    return surface_tension_mn_m * 1.0e-3


def _get_density_kg_m3(df: pd.DataFrame) -> pd.Series:
    """Return liquid density in kg/m^3."""
    return _series_or_default(df, "Density (kg/m^3)", DEFAULT_DENSITY_KG_M3).clip(lower=1e-9)


def _get_evaporation_rate_m_s(df: pd.DataFrame) -> pd.Series:
    """Return solvent evaporation rate in m/s."""
    return _series_or_default(
        df,
        "Evaporation Rate (m/s)",
        DEFAULT_EVAPORATION_RATE_M_S,
    ).clip(lower=1e-12)


def _get_film_width_m(df: pd.DataFrame) -> pd.Series:
    """Return coated width in meters."""
    return _series_or_default(df, "Film Width (m)", DEFAULT_FILM_WIDTH_M).clip(lower=1e-9)


def bonded_layer_adsorption_model(df: pd.DataFrame) -> pd.Series:
    """
    Bonded-layer adsorption model.

    Literature basis:
    - Merzlikine et al. (2004/2005) used an exponential adsorption-growth form.

    Implemented form:
    h_B(t) = h_eq * [1 - exp(-t / tau)]

    Assumption:
    - If no dwell-time column exists, a representative 2000 s immersion is used.
    """
    dwell_time_s = _get_dwell_time(df)
    equilibrium_bonded_nm = 0.80
    tau_s = 1000.0
    prediction = equilibrium_bonded_nm * (1.0 - np.exp(-dwell_time_s / tau_s))
    return prediction.astype(float)


def concentration_dependent_adsorption_time_model(df: pd.DataFrame) -> pd.Series:
    """
    Concentration-dependent adsorption time model.

    Literature basis:
    - Dahiru and Li (2026) pseudo-first-order reversible adsorption framework.

    Implemented form:
    tau(C) = 1 / (k1_eff * C + k2)
    h_B(t) = y0 + A0 * [1 - exp(-t / tau(C))]

    Assumptions:
    - The project concentration values are used directly as the concentration term.
    - If no dwell-time column exists, a representative 2000 s immersion is used.
    """
    concentration = _get_concentration(df)
    dwell_time_s = _get_dwell_time(df)

    y0_nm = 0.40
    a0_nm = 0.40
    k1_eff = 4.3e-4
    k2 = 2.6e-4

    tau_s = 1.0 / (k1_eff * concentration + k2)
    prediction = y0_nm + a0_nm * (1.0 - np.exp(-dwell_time_s / tau_s))
    return prediction.astype(float)


def landau_levich_wet_mobile_layer_model(df: pd.DataFrame) -> pd.Series:
    """
    Landau-Levich wet/mobile-layer model.

    Literature basis:
    - Landau-Levich / Landau-Levich-Derjaguin wet-film entrainment law.

    Implemented form:
    h_LL = 0.94 * (eta * U)^(2/3) / [gamma^(1/6) * (rho * g)^(1/2)]

    Because the experimental target is bonded thickness rather than wet-film
    thickness, the wet-film result is converted to a retained bonded-thickness
    proxy using concentration scaling and a small retention factor.
    """
    concentration = _get_proxy_concentration(df)
    viscosity_pa_s = _get_viscosity_pa_s(df)
    withdrawal_speed_m_s = _get_withdrawal_speed_m_s(df)
    surface_tension_n_m = _get_surface_tension_n_m(df)
    density_kg_m3 = _get_density_kg_m3(df)

    wet_film_m = 0.94 * (viscosity_pa_s * withdrawal_speed_m_s) ** (2.0 / 3.0)
    wet_film_m /= (surface_tension_n_m ** (1.0 / 6.0)) * np.sqrt(density_kg_m3 * G)

    wet_film_nm = wet_film_m * NM_PER_M
    prediction = wet_film_nm * concentration * DEFAULT_WET_TO_BONDED_RETENTION
    return prediction.astype(float)


def capillarity_evaporation_regime_model(df: pd.DataFrame) -> pd.Series:
    """
    Capillarity / evaporation regime model.

    Literature basis:
    - Chapter 10 / Grosso-Faustini capillarity-regime expression.

    Implemented form:
    h_cap = k_i * E / (L * U)

    For direct use on the PDMS table, concentration is used as the material term
    inside k_i so that the model remains row-varying without requiring extra
    process columns.
    """
    concentration = _get_proxy_concentration(df)
    evaporation_rate_m_s = _get_evaporation_rate_m_s(df)
    film_width_m = _get_film_width_m(df)
    withdrawal_speed_m_s = _get_withdrawal_speed_m_s(df)

    material_constant_nm = 800.0
    prediction = material_constant_nm * concentration * evaporation_rate_m_s / (film_width_m * withdrawal_speed_m_s)
    return prediction.astype(float)


def combined_capillarity_landau_levich_model(df: pd.DataFrame) -> pd.Series:
    """
    Combined capillarity + Landau-Levich model.

    Literature basis:
    - h_f = k_i * E / (L * U) + D * U^(2/3)

    As with the pure capillarity model, concentration is used as the row-varying
    material term so the model can run on the available dataset.
    """
    concentration = _get_proxy_concentration(df)
    evaporation_rate_m_s = _get_evaporation_rate_m_s(df)
    film_width_m = _get_film_width_m(df)
    withdrawal_speed_m_s = _get_withdrawal_speed_m_s(df)

    capillary_term_nm = 800.0 * concentration * evaporation_rate_m_s / (film_width_m * withdrawal_speed_m_s)
    draining_term_nm = 120.0 * concentration * np.power(withdrawal_speed_m_s, 2.0 / 3.0)
    prediction = capillary_term_nm + draining_term_nm
    return prediction.astype(float)


def get_analytical_models() -> list[AnalyticalModel]:
    """Return analytical models that should participate in the comparison."""
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
            description="Capillary-feed / evaporation-controlled thickness proxy.",
            predictor=capillarity_evaporation_regime_model,
        ),
        AnalyticalModel(
            name="Combined Capillarity + Landau-Levich",
            description="Superposed capillarity and draining contributions.",
            predictor=combined_capillarity_landau_levich_model,
        ),
    ]
