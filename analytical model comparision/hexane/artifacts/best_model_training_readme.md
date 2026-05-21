# Bayesian RF Hyperparameter Optimization

Training date: 2026-05-10T10:26:21.892729

## Configuration

- Dataset: `Holdout Validation/Data/synthetic_data_improved.csv`
- Samples: 3000
- Features: 7
- CV folds: 5
- Stratification bins: 10
- Bayesian iterations: 32

## Performance

- Baseline RF OOF R2: 0.971620
- Optimized RF OOF R2: 0.973424
- Optimized RF best CV R2: 0.973392
- Full-fit R2 on all 3000 samples: 0.996519

## Best Parameters

- `bootstrap`: `True`
- `max_depth`: `30`
- `max_features`: `log2`
- `min_samples_leaf`: `1`
- `min_samples_split`: `2`
- `n_estimators`: `323`
- `random_state`: `42`
- `n_jobs`: `1`

## Outputs

- `summary_metrics.json`
- `bayes_search_results.csv`
- `out_of_fold_predictions.csv`
- `feature_importances.csv`
- `best_estimator.pkl`
- `optimization_summary.png`