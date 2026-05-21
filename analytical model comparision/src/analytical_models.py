"""Registry for analytical thickness models used in the comparison runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


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


def get_analytical_models() -> list[AnalyticalModel]:
    """Return analytical models that should participate in the comparison."""
    return []


def analytical_model_template(df: pd.DataFrame) -> pd.Series:
    """
    Template for future analytical formulas.

    Replace this function with the formula from a paper and register it inside
    ``get_analytical_models`` once the required variables are available.
    """
    raise NotImplementedError("Analytical formula not implemented yet.")
