# Data

This folder stores datasets and documentation for thin-film ML experiments.

## Datasets

- `agg.data.xlsx`: Main aggregated dataset used for model training and evaluation.
- `placeholder_film_data.csv`: Legacy placeholder data (for reference only).
- `Hexane+PDMS-new.xlsx`: Additional experimental data.
- `Hexane_PDMS_ML_Training_Dataset.xlsx`: Training dataset archive.

## Data Processing

The `agg.data.xlsx` dataset is automatically processed by the model pipelines:
- **Feature columns**: All columns except those containing 'thickness', 'film', or 'thin' in their names.
- **Target columns**: All columns containing 'thickness' or 'film' in their names.

## Model Evaluation

All models now generate:
- **Evaluation Metrics**: MSE, RMSE, MAE, R² Score
- **Visualizations**: 
  - Predictions vs Actual scatter plot
  - Residual analysis plot
  - Residual distribution histogram
  - Performance metrics summary

Outputs are saved to each model's `results/` directory.
