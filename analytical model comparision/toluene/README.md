# Toluene Analytical Comparison

This folder runs the same experimental comparison workflow as the hexane folder, but against the local [toluene+pdms.csv](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/toluene/toluene+pdms.csv) dataset.

- The best bonded Bayesian-optimized RF artifact is reused from the project-wide best model.
- The optimized mobile-layer RF is reconstructed from its committed search summary.
- Toluene solvent constants are injected for the ML feature frame.
- Bonded metrics contain only the RF and adsorption-motivated analytical models; Landau--Levich is excluded.
- Mobile metrics compare `Total Thickness - Bonded Thickness` with the mobile RF and the Landau--Levich dry-mobile estimate.
- The evaporation-dependent analytical models remain symbolic in terms of `E`.

Run:

```bash
python "Holdout Validation/src/optimize_rf_mobile_layer_bayesian.py"
python "analytical model comparision/toluene/src/run_comparison.py"
```
