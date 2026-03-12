# Optimized Hyperparameter Training Results

This directory contains results from training models with OPTIMIZED hyperparameters.

## Optimization Details

Optimization was performed using GridSearchCV (BPNN, KNN) and RandomizedSearchCV (RFR).
See `src/optimization_results/` for detailed optimization logs.

## Models Trained

1. **BPNN** - Backpropagation Neural Networks
2. **KNN** - K-Nearest Neighbors
3. **Linear Regression** - Linear Model
4. **Random Forest Regression** - Ensemble Model

## Results Summary

### BPNN
- R² Score: 0.798358
- RMSE: 0.041920
- MAE: 0.031125
- MSE: 0.001757

### KNN
- R² Score: 0.797171
- RMSE: 0.042043
- MAE: 0.027929
- MSE: 0.001768

### Linear Regression
- R² Score: 0.739808
- RMSE: 0.047619
- MAE: 0.039922
- MSE: 0.002268

### Random Forest Regression
- R² Score: 0.808200
- RMSE: 0.040884
- MAE: 0.027259
- MSE: 0.001672

