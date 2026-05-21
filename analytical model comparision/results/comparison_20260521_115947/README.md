# Comparison Run

Created: 2026-05-21T11:59:47.532880

## Inputs

- Experimental dataset: `C:\Users\alexg\Downloads\Pitt-ML-MSE-Project\models\feature engineering v1\data FE-V1\agg.data.xlsx`
- Best model artifact: `C:\Users\alexg\Downloads\Pitt-ML-MSE-Project\analytical model comparision\artifacts\best_estimator.pkl`
- Rows evaluated: 58
- Analytical models are currently configured for `PDMS` in `hexane`.
- Concentration is read directly from `agg.data.xlsx`.
- Dwell time, withdrawal speed, film width, evaporation rate, and density are fixed experiment-level constants in the current implementation.
- The current evaporation-rate value is a temporary placeholder and not yet tied to a cited source.
- Hexane relative evaporation reference: `https://www.stenutz.eu/chem/evaporation.php?s=3`.
- That source provides relative evaporation behavior and vapor pressure, but not the effective model evaporation rate in m/s.
- The density used in the Landau-Levich term is the coating-solution density, currently approximated by hexane for the dilute PDMS + hexane bath.

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