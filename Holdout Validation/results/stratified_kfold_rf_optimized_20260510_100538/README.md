# Stratified K-Fold Cross-Validation Results

Training date: 2026-05-10T10:05:38.132280

## Configuration

- Dataset: `Holdout Validation/Data/synthetic_data_improved.csv`
- Samples: 3000
- Features: 7
- Target: `Bonded Thickness (nm)`
- Folds: 5
- Stratification bins: 10
- Stratification method: quantile bins over the continuous target
- Model source: `Holdout Validation/models/pipeline_rf_optimized.py`

## Aggregate Performance

- Mean R2: 0.971593 +/- 0.003533
- Mean RMSE: 0.030808 +/- 0.001807
- Mean MAE: 0.019663 +/- 0.000911
- Out-of-fold R2: 0.971620
- Out-of-fold RMSE: 0.030851

## Outputs

- `summary_metrics.json`
- `fold_metrics.csv` and `fold_metrics.json`
- `out_of_fold_predictions.csv`
- `feature_importances_by_fold.csv`
- `feature_importances_summary.csv`
- `stratified_kfold_rf_optimized_summary.png`