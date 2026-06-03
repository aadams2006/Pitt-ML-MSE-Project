# Comparison Run

Created: 2026-06-03T09:14:35.436659

## Inputs

- Experimental dataset: `C:\Users\alexg\Downloads\Pitt-ML-MSE-Project\analytical model comparision\ethyl acetate\ethyl acetate+pdms.csv`
- Best model artifact: `C:\Users\alexg\Downloads\Pitt-ML-MSE-Project\analytical model comparision\ethyl acetate\artifacts\best_estimator.pkl`
- Rows evaluated: 33
- Analytical models are currently configured for `PDMS` in `ethyl acetate`.
- Concentration is read directly from the solvent subfolder CSV.
- Dwell time, withdrawal speed, film width, and density are fixed experiment-level constants in the current implementation.
- The density used in the Landau-Levich term is the coating-solution density, currently approximated by ethyl acetate for the dilute PDMS + ethyl acetate bath.

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