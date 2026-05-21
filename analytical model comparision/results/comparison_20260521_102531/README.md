# Comparison Run

Created: 2026-05-21T10:25:32.531245

## Inputs

- Experimental dataset: `C:\Users\alexg\Downloads\Pitt-ML-MSE-Project\models\feature engineering v1\data FE-V1\agg.data.xlsx`
- Best model artifact: `C:\Users\alexg\Downloads\Pitt-ML-MSE-Project\analytical model comparision\artifacts\best_estimator.pkl`
- Rows evaluated: 58
- Analytical models are currently configured for `PDMS` in `hexane`.
- Concentration is read directly from `agg.data.xlsx`.
- Dwell time, withdrawal speed, film width, evaporation rate, and density are fixed experiment-level constants in the current implementation.
- The evaporation-rate term is a solvent-based effective estimate, not a direct lab measurement.

## Models Included

- `Bayesian Optimized RF`
- `Bonded-Layer Adsorption`
- `Concentration-Dependent Adsorption Time`
- `Landau-Levich Wet/Mobile Layer`
- `Capillarity / Evaporation Regime`
- `Combined Capillarity + Landau-Levich`

## Outputs

- `comparison_predictions.csv`
- `comparison_metrics.csv`
- `comparison_metadata.json`
- `comparison_plot.png`