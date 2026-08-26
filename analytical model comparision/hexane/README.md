# Analytical Model Comparision

This folder is the comparison workspace for the best machine-learning model from the project and the analytical coating models derived from the reference papers.

## Current ML Baselines

- Bonded target: Bayesian-optimized Random Forest, with `R2 = 0.973424` out of fold on the 3000-row synthetic dataset.
- Mobile target: Bayesian-optimized Random Forest, with `R2 = 0.999597` out of fold on 2887 physically valid synthetic rows.
- The mobile estimator is reconstructed from the committed best parameters and cleaned training-row policy rather than stored as another large pickle.

## Evaluation Dataset

- Experimental file: [agg.data.xlsx](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/models/feature%20engineering%20v1/data%20FE-V1/agg.data.xlsx)
- Shape: 58 rows x 4 columns
- Columns:
  - `Concentration (g/mL)`
  - `Uncoated Layer (nm)`
  - `Total Thickness (nm)`
  - `Bonded Thickness (nm)`

## Important Assumptions

The current experimental table is the original hexane dataset and only contains the 3 process/thickness inputs. The best Random Forest was trained on 7 inputs, so the comparison runner augments the experimental rows with the constant hexane solvent properties already used in the synthetic training dataset:

- `Polarity (XLogP3) = 3.9`
- `Viscosity (cP) = 0.377`
- `Boiling Point (K) = 342.039`
- `Surface Tension (mN/m) = 17.89`

For the analytical models:

- The source workbooks identify concentration as g/L; derived tables retain the legacy `Concentration (g/mL)` header.
- Bonded adsorption comparators use the derived table scale directly. The Landau--Levich dry-mobile conversion explicitly converts the source g/L scale to g/mL.
- the missing process terms are treated as fixed experiment-level constants for the `PDMS + hexane` experiment
- confirmed lab constants currently used:
  - `dwell time = 20 s`
  - `withdrawal speed = 1.0 mm/s`
- hexane relative evaporation source: `USDA`, with `Evaporation Rate (BuAc = 1): 9`
- that relative evaporation value supports that hexane is fast-evaporating, but it does not directly provide the effective model evaporation rate `E` in `m/s`
- the density used in the Landau-Levich wet-film term is the coating-solution density, currently approximated by hexane for the dilute `PDMS + hexane` bath

## Target-Separated Analytical Models

- Bonded thickness: `Bonded-Layer Adsorption` and `Concentration-Dependent Adsorption Time` are evaluated numerically.
- Mobile thickness: `Landau-Levich Mobile Layer` is evaluated against `Total Thickness - Bonded Thickness` and compared with the optimized mobile RF.
- `Landau-Levich Mobile Layer` is intentionally absent from bonded-thickness metrics.
- `Capillarity / Evaporation Regime` remains symbolic in terms of `E`; the mixed-regime equation is literature background only and is not registered as a bonded comparator.

The two evaporation-dependent models are intentionally left symbolic until an effective evaporation rate is available for the experiment.

## Validation Caveat

`agg.data.xlsx` was not used as a direct 58-row training table for the final RF artifact, but the final RF was trained on synthetic data generated from this experimental source plus related solvent-expanded datasets. That means this comparison is useful for experimental alignment, but it is not a fully independent external validation.

## Files

- [src/run_comparison.py](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/src/run_comparison.py): Main comparison runner
- [src/analytical_models.py](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/src/analytical_models.py): Registry for analytical formulas
- [results](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/results): Output directory for metrics, predictions, and symbolic-model notes

## Usage

Run:

```bash
python "Holdout Validation/src/optimize_rf_mobile_layer_bayesian.py"
python "analytical model comparision/hexane/src/run_comparison.py"
```

The script will:

- load the saved best Random Forest model
- evaluate it on `agg.data.xlsx`
- evaluate the bonded ML and adsorption-model comparison without Landau--Levich
- reconstruct the optimized mobile RF and compare it with Landau--Levich on the derived mobile target
- record the evaporation-dependent analytical models symbolically in terms of `E`
- save separate bonded and mobile predictions, metrics, and plots under `results/`

## Next Step

When you obtain an effective evaporation rate `E`, the two symbolic models can be switched from formula-only reporting to full numeric evaluation.
