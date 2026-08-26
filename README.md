# Pitt ML MSE Project

This repository is organized as a machine-learning workspace for thin-film processing prediction. It provides a consistent structure for experimenting with multiple model families while sharing the aggregated dataset.

The current analytical-comparison workflow keeps the physical targets separate: adsorption-motivated models are evaluated against bonded thickness, while Landau--Levich and a separately Bayesian-optimized random forest are evaluated against the mobile layer (`total - bonded`). See `analytical model comparision/` and `Holdout Validation/src/optimize_rf_mobile_layer_bayesian.py` for the reproducible workflows and committed results.

## Repository layout

- `data/`: Dataset files and data documentation.
  - `agg.data.xlsx`: Primary aggregated dataset for training and evaluation.
- `models/`: Model-specific experiment folders with complete pipelines.
  - `bpnn/`: Backpropagation Neural Networks (BPNNs).
  - `knn/`: K-Nearest Neighbors (KNN).
  - `linear_regression/`: Linear Regression.
  - `rfr/`: Random Forest Regression (RFR).

## Features

Each model pipeline includes:
- **Data loading**: Automatic loading of `agg.data.xlsx` with intelligent feature/target column detection
- **Model training**: Train/test split with standard preprocessing
- **Evaluation metrics**: MSE, RMSE, MAE, R² Score
- **Visualizations**: 
  - Predictions vs Actual scatter plots
  - Residual analysis and distribution
  - Performance metrics summary cards
- **Output management**: Results saved to `model/results/` directory

## Usage

Each model folder contains a `pipeline.py` with the following functions:
- `load_agg_data()`: Load the aggregated dataset
- `split_features_targets()`: Separate features from targets
- `train_and_evaluate()`: Train model and compute metrics
- `plot_results()`: Generate and save evaluation visualizations
