# PI Next Steps Implementation — 2026-08-26

This note records the implementation of the three follow-up items from the PI meeting:

1. expand the paper's analytical-model literature review and background;
2. train a Bayesian-optimized Random Forest on the mobile/wet-layer target and compare it with Landau--Levich; and
3. remove Landau--Levich from bonded-thickness prediction.

## Target definitions

- Bonded target: `Bonded Thickness (nm)`.
- Mobile target: `Mobile Layer (nm) = Total Thickness (nm) - Bonded Thickness (nm)`.
- The experimental comparison clips only negative derived differences to zero. This affects one toluene row whose rounded measurements differ by `-0.01 nm`; the audit is saved with the comparison.
- The synthetic training run excludes negative derived mobile values rather than clipping them. The tracked source dataset is unchanged.

## Mobile-layer Bayesian optimization

Runner: `Holdout Validation/src/optimize_rf_mobile_layer_bayesian.py`

Training source: `models/feature engineering v1/data FE-V1/synthetic_data_improved.csv`

Data audit:

- input rows: 3000;
- missing rows: 0;
- negative derived mobile rows excluded: 113;
- rows used: 2887.

Features:

- `Polarity (XLogP3)`;
- `Viscosity (cP)`;
- `Boiling Point (K)`;
- `Surface Tension (mN/m)`;
- legacy `Concentration (g/mL)` column;
- `Uncoated Layer (nm)`;
- `Total Thickness (nm)`.

`Bonded Thickness (nm)` is used only to derive the response and is not an RF feature.

Validation and search:

- 32 `BayesSearchCV` iterations;
- five folds stratified on ten mobile-target quantile bins;
- scoring: `R2`;
- random seed: 42;
- the same documented folds are used for baseline and optimized out-of-fold predictions.

Best parameters:

| Parameter | Value |
|---|---:|
| `bootstrap` | `True` |
| `max_depth` | 27 |
| `max_features` | `None` |
| `min_samples_leaf` | 1 |
| `min_samples_split` | 2 |
| `n_estimators` | 450 |
| `random_state` | 42 |
| `n_jobs` | 1 |

Performance:

| Model | OOF R2 | OOF RMSE (nm) | OOF MAE (nm) |
|---|---:|---:|---:|
| Baseline RF | 0.999350 | 0.046171 | 0.026888 |
| Bayesian-optimized RF | 0.999597 | 0.036338 | 0.020013 |

Results are committed under `Holdout Validation/results/rf_bayesian_optimization_mobile_layer_20260826_104905/`. The folder includes the full search table, feature importance, out-of-fold predictions, metrics, README, and summary plot. A second large model pickle is intentionally not committed; comparison runners reconstruct the deterministic estimator from the recorded parameters and audited training rows.

## Landau--Levich mobile-layer implementation

The analytical prediction starts from the wet-solution thickness:

`h_LLD = 0.94 (mu U)^(2/3) / [gamma^(1/6) (rho g)^(1/2)]`.

The source workbooks label concentration in g/L, although derived tables retain the legacy header `Concentration (g/mL)`. The code converts the numeric source scale to g/mL and approximates the PDMS volume fraction as:

`phi_PDMS = c_g_per_mL / (c_g_per_mL + 0.965 g/mL)`.

The comparison prediction is `h_mobile = phi_PDMS * h_LLD`. It uses the recorded solvent constants and a fixed withdrawal speed of `1.0 mm/s`. It is not fitted to bonded thickness and has no empirical retention factor.

## Experimental-data comparison

| Solvent | Model | R2 | RMSE (nm) | MAE (nm) |
|---|---|---:|---:|---:|
| Hexane | Bayesian-optimized RF | 0.999986 | 0.010386 | 0.006432 |
| Hexane | Landau--Levich mobile layer | 0.990529 | 0.267725 | 0.211779 |
| Toluene | Bayesian-optimized RF | 0.930008 | 0.026170 | 0.011328 |
| Toluene | Landau--Levich mobile layer | -95254.722098 | 30.529349 | 15.309791 |
| Ethyl acetate | Bayesian-optimized RF | 0.999790 | 0.009323 | 0.006729 |
| Ethyl acetate | Landau--Levich mobile layer | -18.186017 | 2.821235 | 1.654359 |

The poor toluene and ethyl acetate analytical results are retained rather than recalibrated away. They show that the fixed-property entrainment-plus-volume-fraction approximation is insufficient for those systems. The extreme toluene `R2` is amplified by its narrow measured target range (`0–0.47 nm`), while its RMSE independently shows the scale error.

## Bonded-thickness correction

- `get_bonded_layer_models()` excludes Landau--Levich in all three solvent modules.
- The previous fitted `landau_levich_wet_mobile_layer` blocks were removed from all three analytical parameter JSON files.
- The analytical fitting scripts no longer fit a Landau--Levich retention proxy to bonded thickness.
- Existing bonded output filenames are retained for compatibility, but their metrics and plots do not contain Landau--Levich.
- New `mobile_layer_predictions.csv`, `mobile_layer_metrics.csv`, and `mobile_layer_comparison_plot.png` outputs keep the second target separate.

## Paper changes

Both `research_paper.tex` and `overleaf_paper_bundle/research_paper.tex` now include:

- expanded Landau--Levich--Derjaguin assumptions and validity discussion;
- explicit bonded/mobile target separation based on adsorption and viscous-flow literature;
- adsorption-kinetics background for bonded models;
- evaporation, capillary-feed, drainage, and mixed-regime background;
- the mobile-layer training and analytical-conversion methodology;
- the full solvent comparison table; and
- limitations covering synthetic validity, state-variable coupling, concentration units, fixed analytical properties, and lack of fully independent validation.

## Reproduction commands

From the repository root:

```bash
python "Holdout Validation/src/optimize_rf_mobile_layer_bayesian.py"
python "analytical model comparision/hexane/src/run_comparison.py"
python "analytical model comparision/toluene/src/run_comparison.py"
python "analytical model comparision/ethyl acetate/src/run_comparison.py"
```

Install project dependencies first with `pip install -r requirements.txt`; `scikit-optimize` is now declared explicitly.

## Interpretation limits

- The solvent-specific experimental tables informed the synthetic generator, so the experimental comparisons are alignment checks rather than fully independent external validation.
- Total thickness is both an RF feature and a term in the derived mobile target. The score is valid for a workflow where total thickness is already measured, but it is not evidence of purely process-variable forecasting.
- Solution viscosity, surface tension, and density are approximated with fixed solvent values, and effective local evaporation is unavailable.
- The capillary-only comparator remains symbolic until an effective evaporation rate and coating-zone width are measured. The mixed capillary/entrainment equation is discussed as literature background but is not registered in the bonded comparison.
