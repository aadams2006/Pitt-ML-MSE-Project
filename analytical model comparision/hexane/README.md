# Analytical Model Comparision

This folder is the comparison workspace for the best machine-learning model from the project and the analytical coating models derived from the reference papers.

## Current ML Baseline

- Best model: Bayesian-optimized Random Forest
- Source artifact: [artifacts/best_estimator.pkl](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/artifacts/best_estimator.pkl)
- Source metrics: [artifacts/summary_metrics.json](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/artifacts/summary_metrics.json)
- Best out-of-fold performance on the 3000-sample synthetic training dataset: `R2 = 0.973424`

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

- `Concentration (g/mL)` is read directly from `agg.data.xlsx`
- the missing process terms are treated as fixed experiment-level constants for the `PDMS + hexane` experiment
- confirmed lab constants currently used:
  - `dwell time = 2000 s`
  - `withdrawal speed = 1.0 mm/s`
- hexane relative evaporation source: `USDA`, with `Evaporation Rate (BuAc = 1): 9`
- that relative evaporation value supports that hexane is fast-evaporating, but it does not directly provide the effective model evaporation rate `E` in `m/s`
- the density used in the Landau-Levich wet-film term is the coating-solution density, currently approximated by hexane for the dilute `PDMS + hexane` bath

## Analytical Models

- `Bonded-Layer Adsorption`: evaluated numerically
- `Concentration-Dependent Adsorption Time`: evaluated numerically
- `Landau-Levich Wet/Mobile Layer`: evaluated numerically
- `Capillarity / Evaporation Regime`: kept symbolic in terms of `E`
- `Combined Capillarity + Landau-Levich`: kept symbolic in terms of `E`

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
python "analytical model comparision/src/run_comparison.py"
```

The script will:

- load the saved best Random Forest model
- evaluate it on `agg.data.xlsx`
- evaluate the analytical models that can be computed numerically from the available constants
- record the evaporation-dependent analytical models symbolically in terms of `E`
- save side-by-side predictions and metrics under `results/`

## Next Step

When you obtain an effective evaporation rate `E`, the two symbolic models can be switched from formula-only reporting to full numeric evaluation.
