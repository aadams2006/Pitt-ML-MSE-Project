# Comparison Run

Created: 2026-06-04T11:31:39.080514

## Inputs

- Experimental dataset: `C:\Users\alexg\Downloads\Pitt-ML-MSE-Project\models\feature engineering v1\data FE-V1\agg.data.xlsx`
- Best model artifact: `C:\Users\alexg\Downloads\Pitt-ML-MSE-Project\analytical model comparision\hexane\artifacts\best_estimator.pkl`
- Rows evaluated: 58
- Analytical models are currently configured for `PDMS` in `hexane`.
- Concentration is read directly from `agg.data.xlsx`.
- Dwell time, withdrawal speed, film width, and density are fixed experiment-level constants in the current implementation.
- Hexane relative evaporation reference: `USDA`, with `BuAc = 1 -> 9.0`.
- That relative evaporation value does not provide the effective model evaporation rate in m/s.
- The density used in the Landau-Levich term is the coating-solution density, currently approximated by hexane for the dilute PDMS + hexane bath.

## Models Included

- `Bayesian Optimized RF`
- `Bonded-Layer Adsorption`
- `Concentration-Dependent Adsorption Time`
- `Landau-Levich Wet/Mobile Layer`
- `Capillarity / Evaporation Regime`: symbolic only, requires `E`
- `Combined Capillarity + Landau-Levich`: symbolic only, requires `E`

## Symbolic Models

- `Capillarity / Evaporation Regime`: `h_cap = k_i E / (L U)`
- `Combined Capillarity + Landau-Levich`: `h_f = k_i E / (L U) + D U^(2/3)`

## Outputs

- `comparison_predictions.csv`
- `comparison_metrics.csv`
- `comparison_metadata.json`
- `comparison_plot.png`