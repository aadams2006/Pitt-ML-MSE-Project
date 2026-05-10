# RF + GB Ensemble Evaluation

Training date: 2026-05-10T10:15:16.626983

## Configuration

- Dataset: `Holdout Validation/Data/synthetic_data_improved.csv`
- Samples: 3000
- Features: 7
- Folds: 5
- Inner folds for blend/stacking selection: 5
- Models evaluated: Random Forest, Gradient Boosting, 50/50 blend, adaptive blend, stacked ridge

## Best Model

- Best by out-of-fold R2: Stacked Ridge

## Out-of-Fold R2

- Stacked Ridge: 0.972202
- Adaptive Blend: 0.971985
- Fixed Blend 50/50: 0.971920
- Random Forest: 0.971620
- Gradient Boosting: 0.969843

## Outputs

- `summary_metrics.json` and `summary_metrics.csv`
- `fold_metrics_by_model.csv`
- `out_of_fold_summary.csv`
- `out_of_fold_predictions_by_model.csv`
- `feature_importances_by_fold.csv` and `feature_importances_summary.csv`
- `adaptive_blend_weights_by_fold.csv`
- `stacking_coefficients_by_fold.csv`
- `feature_transformation_candidates.csv`
- `ensemble_comparison_summary.png`