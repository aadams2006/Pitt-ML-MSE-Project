# Comparison Run

Created: 2026-08-26T10:49:26.275309

## Inputs

- Experimental dataset: `/workspace/scratch/ee3524fcd8af/pitt-ml-mse/models/feature engineering v1/data FE-V1/agg.data.xlsx`
- Best model artifact: `/workspace/scratch/ee3524fcd8af/pitt-ml-mse/analytical model comparision/hexane/artifacts/best_estimator.pkl`
- Rows evaluated: 58
- Analytical models are currently configured for `PDMS` in `hexane`.
- Concentration is read directly from `agg.data.xlsx`.
- Dwell time, withdrawal speed, film width, and density are fixed experiment-level constants in the current implementation.
- Hexane relative evaporation reference: `USDA`, with `BuAc = 1 -> 9.0`.
- That relative evaporation value does not provide the effective model evaporation rate in m/s.
- The density used in the Landau-Levich term is the coating-solution density, currently approximated by hexane for the dilute PDMS + hexane bath.

## Bonded-Thickness Models Included

- `Bayesian Optimized RF`
- `Bonded-Layer Adsorption`
- `Concentration-Dependent Adsorption Time`
- `Capillarity / Evaporation Regime`: symbolic only, requires `E`

## Symbolic Models

- `Capillarity / Evaporation Regime`: `h_cap = k_i E / (L U)`

Landau--Levich is intentionally excluded from this bonded-thickness comparison.

## Mobile-Layer Comparison

- Target: `Mobile Layer (nm) = Total Thickness (nm) - Bonded Thickness (nm)`
- Models: `Bayesian Optimized RF (Mobile Layer)` and `Landau-Levich Mobile Layer`
- Mobile RF optimization summary: `/workspace/scratch/ee3524fcd8af/pitt-ml-mse/Holdout Validation/results/rf_bayesian_optimization_mobile_layer_20260826_104905/summary_metrics.json`
- Negative experimental differences clipped to zero: 0
- Landau--Levich wet thickness is converted to dry mobile PDMS thickness with the PDMS solution volume fraction; it is not fit to bonded thickness.

## Outputs

- `comparison_predictions.csv` (bonded thickness)
- `comparison_metrics.csv` (bonded thickness)
- `comparison_plot.png` (bonded thickness)
- `mobile_layer_predictions.csv`
- `mobile_layer_metrics.csv`
- `mobile_layer_comparison_plot.png`
- `comparison_metadata.json`
