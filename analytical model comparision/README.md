# Analytical Model Comparision

This folder is the comparison workspace for the best machine-learning model from the project and the analytical coating models that will be added from the reference papers.

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

## Important Assumption

The current experimental table is the original hexane dataset and only contains the 3 process/thickness inputs. The best Random Forest was trained on 7 inputs, so the comparison runner augments the experimental rows with the constant hexane solvent properties already used in the synthetic training dataset:

- `Polarity (XLogP3) = 3.9`
- `Viscosity (cP) = 0.377`
- `Boiling Point (K) = 342.039`
- `Surface Tension (mN/m) = 17.89`

For the analytical models:

- `Concentration (g/mL)` is read directly from `agg.data.xlsx`
- the process constants that are not in the dataset are currently treated as fixed experiment-level constants for the `PDMS + hexane` experiment
- the current evaporation-rate value is a temporary placeholder and is not yet tied to a cited source
- the density used in the Landau-Levich wet-film term is the coating-solution density, currently approximated by hexane for the dilute `PDMS + hexane` bath

## Validation Caveat

`agg.data.xlsx` was not used as a direct 58-row training table for the final RF artifact, but the final RF was trained on synthetic data generated from this experimental source plus related solvent-expanded datasets. That means this comparison is useful for experimental alignment, but it is not a fully independent external validation.

## Files

- [src/run_comparison.py](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/src/run_comparison.py): Main comparison runner
- [src/analytical_models.py](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/src/analytical_models.py): Registry for analytical formulas
- [results](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/results): Output directory for metrics, predictions, and plots

## Implemented Analytical Models

- `Bonded-Layer Adsorption`
- `Concentration-Dependent Adsorption Time`
- `Landau-Levich Wet/Mobile Layer`
- `Capillarity / Evaporation Regime`
- `Combined Capillarity + Landau-Levich`

These are implemented as executable literature-inspired formulas with fallback assumptions for missing variables such as dwell time, withdrawal speed, coated width, evaporation rate, and density.

## Usage

Run:

```bash
python "analytical model comparision/src/run_comparison.py"
```

The script will:

- load the saved best Random Forest model,
- evaluate it on `agg.data.xlsx`,
- evaluate any analytical models registered in `src/analytical_models.py`,
- save side-by-side predictions and metrics under `results/`.

## Next Step For Analytical Formulas

When you send the papers/formulas, I only need to add them inside `src/analytical_models.py`. The runner is already set up to score them against the same experimental rows and compare them directly to the ML model.
