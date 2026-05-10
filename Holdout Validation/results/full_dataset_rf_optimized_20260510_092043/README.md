# Full-Dataset Random Forest Retrain

Training date: 2026-05-10T09:20:43.187109

## Configuration

- Dataset: `Holdout Validation/Data/synthetic_data_improved.csv`
- Samples: 3000
- Features: 7
- Target: `Bonded Thickness (nm)`
- Model source: `Holdout Validation/models/pipeline_rf_optimized.py`

## Metrics

- Training R2: 0.990426
- OOB R2: 0.972971
- RMSE: 0.017919
- MAE: 0.011265
- MSE: 0.000321

## Outputs

- `metrics.json`
- `training_predictions.csv`
- `feature_importances.csv`
- `random_forest_optimized_full_dataset.pkl`
- `random_forest_optimized_full_dataset_training.png`