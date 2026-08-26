# Bayesian RF Optimization: Mobile Layer

Training date: 2026-08-26T10:49:05.243019

## Target and data policy

- Target: `Mobile Layer (nm) = Total Thickness (nm) - Bonded Thickness (nm)`
- Source dataset: `models/feature engineering v1/data FE-V1/synthetic_data_improved.csv`
- Input rows: 3000
- Negative mobile-layer rows excluded: 113
- Rows used for training: 2887
- The source CSV was not modified.

## Validation and performance

- Bayesian iterations: 32
- Stratified folds: 5
- Baseline RF OOF R2: 0.999350
- Optimized RF OOF R2: 0.999597
- Optimized RF OOF RMSE: 0.036338 nm
- Best search CV R2: 0.999567

## Best parameters

- `bootstrap`: `True`
- `max_depth`: `27`
- `max_features`: `None`
- `min_samples_leaf`: `1`
- `min_samples_split`: `2`
- `n_estimators`: `450`
- `random_state`: `42`
- `n_jobs`: `1`

## Outputs

- `summary_metrics.json`
- `bayes_search_results.csv`
- `out_of_fold_predictions.csv`
- `feature_importances.csv`
- `optimization_summary.png`

The estimator is reproducibly rebuilt from `best_params` by the solvent comparison runners; no duplicate large model binary is committed.