"""Hyperparameter optimization using GridSearchCV for all models."""

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load data
data_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_data_improved.csv"
df = pd.read_csv(data_path)

# Features and targets
FEATURE_COLUMNS = ['Concentration (g/mL)', 'Uncoated Layer (nm)', 'Total Thickness (nm)']
TARGET_COLUMNS = ['Bonded Thickness (nm)']

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMNS].copy().values.ravel()

# Remove any NaN values
valid_idx = ~(X.isna().any(axis=1) | np.isnan(y))
X = X[valid_idx].reset_index(drop=True)
y = y[valid_idx]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create results directory
opt_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
opt_results_dir = Path(__file__).resolve().parent.parent / "src" / "optimization_results"
opt_results_dir.mkdir(parents=True, exist_ok=True)

optimization_log = {
    "optimization_date": datetime.now().isoformat(),
    "data_path": str(data_path),
    "train_shape": X_train_scaled.shape,
    "test_shape": X_test_scaled.shape,
    "models": {}
}

print("="*80)
print("HYPERPARAMETER OPTIMIZATION")
print("="*80)
print(f"Training data shape: {X_train_scaled.shape}")
print(f"Test data shape: {X_test_scaled.shape}\n")

# ============================================================================
# 1. BPNN Optimization
# ============================================================================
print("="*80)
print("Optimizing BPNN (Backpropagation Neural Networks)")
print("="*80)

bpnn_param_grid = {
    'hidden_layer_sizes': [
        (50,), (100,), (150,),
        (50, 25), (100, 50), (150, 75),
        (100, 50, 25), (150, 75, 37)
    ],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'max_iter': [500, 1000, 2000],
    'alpha': [0.00001, 0.0001, 0.001]
}

bpnn = MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1)
bpnn_grid = GridSearchCV(
    bpnn, bpnn_param_grid, cv=5, n_jobs=-1, scoring='r2', verbose=1
)
print("\nSearching parameter space...")
bpnn_grid.fit(X_train_scaled, y_train)

bpnn_best_params = bpnn_grid.best_params_
bpnn_best_score = bpnn_grid.best_score_
bpnn_test_score = bpnn_grid.score(X_test_scaled, y_test)

print(f"\nBest Parameters: {bpnn_best_params}")
print(f"Best Cross-Val R²: {bpnn_best_score:.6f}")
print(f"Test Set R²: {bpnn_test_score:.6f}")

optimization_log["models"]["bpnn"] = {
    "original_params": {
        "hidden_layer_sizes": [100, 50],
        "learning_rate_init": 0.001,
        "max_iter": 1000,
        "alpha": 0.0001
    },
    "optimized_params": bpnn_best_params,
    "cv_r2_score": float(bpnn_best_score),
    "test_r2_score": float(bpnn_test_score)
}

# ============================================================================
# 2. KNN Optimization
# ============================================================================
print("\n" + "="*80)
print("Optimizing KNN (K-Nearest Neighbors)")
print("="*80)

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsRegressor()
knn_grid = GridSearchCV(
    knn, knn_param_grid, cv=5, n_jobs=-1, scoring='r2', verbose=1
)
print("\nSearching parameter space...")
knn_grid.fit(X_train_scaled, y_train)

knn_best_params = knn_grid.best_params_
knn_best_score = knn_grid.best_score_
knn_test_score = knn_grid.score(X_test_scaled, y_test)

print(f"\nBest Parameters: {knn_best_params}")
print(f"Best Cross-Val R²: {knn_best_score:.6f}")
print(f"Test Set R²: {knn_test_score:.6f}")

optimization_log["models"]["knn"] = {
    "original_params": {
        "n_neighbors": 5,
        "weights": "uniform",
        "p": 2,
        "metric": "euclidean"
    },
    "optimized_params": knn_best_params,
    "cv_r2_score": float(knn_best_score),
    "test_r2_score": float(knn_test_score)
}

# ============================================================================
# 3. Linear Regression (Limited Optimization)
# ============================================================================
print("\n" + "="*80)
print("Evaluating Linear Regression")
print("="*80)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_cv_score = lr.score(X_train_scaled, y_train)
lr_test_score = lr.score(X_test_scaled, y_test)

print(f"Training R²: {lr_cv_score:.6f}")
print(f"Test Set R²: {lr_test_score:.6f}")

optimization_log["models"]["linear_regression"] = {
    "original_params": {
        "fit_intercept": True,
        "normalize": False
    },
    "optimized_params": {
        "fit_intercept": True,
        "normalize": False
    },
    "note": "Linear Regression has no hyperparameters to optimize",
    "train_r2_score": float(lr_cv_score),
    "test_r2_score": float(lr_test_score)
}

# ============================================================================
# 4. Random Forest Regression Optimization
# ============================================================================
print("\n" + "="*80)
print("Optimizing Random Forest Regression")
print("="*80)

rfr_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rfr = RandomForestRegressor(random_state=42, n_jobs=-1)
rfr_grid = RandomizedSearchCV(
    rfr, rfr_param_grid, cv=5, n_iter=30, n_jobs=-1, 
    scoring='r2', verbose=1, random_state=42
)
print("\nSearching parameter space (RandomizedSearchCV)...")
rfr_grid.fit(X_train_scaled, y_train)

rfr_best_params = rfr_grid.best_params_
rfr_best_score = rfr_grid.best_score_
rfr_test_score = rfr_grid.score(X_test_scaled, y_test)

print(f"\nBest Parameters: {rfr_best_params}")
print(f"Best Cross-Val R²: {rfr_best_score:.6f}")
print(f"Test Set R²: {rfr_test_score:.6f}")

optimization_log["models"]["rfr"] = {
    "original_params": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },
    "optimized_params": rfr_best_params,
    "cv_r2_score": float(rfr_best_score),
    "test_r2_score": float(rfr_test_score)
}

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("OPTIMIZATION SUMMARY")
print("="*80)

summary_data = []
for model_name, model_info in optimization_log["models"].items():
    original_r2 = "N/A (no optimization)"
    if "test_r2_score" in model_info:
        original_r2 = 0.7561 if model_name == "bpnn" else \
                      0.7987 if model_name == "knn" else \
                      0.7398 if model_name == "linear_regression" else \
                      0.7861
    
    optimized_r2 = model_info.get("test_r2_score", "N/A")
    improvement = "N/A"
    if isinstance(optimized_r2, float) and isinstance(original_r2, float):
        improvement = f"{(optimized_r2 - original_r2):.6f}"
    
    print(f"\n{model_name.upper()}")
    print(f"  Original Test R²: {original_r2}")
    print(f"  Optimized Test R²: {optimized_r2:.6f}")
    print(f"  Improvement: {improvement}")

# Save optimization results
opt_log_path = opt_results_dir / f"optimization_log_{opt_timestamp}.json"
with open(opt_log_path, 'w') as f:
    json.dump(optimization_log, f, indent=2)

print(f"\n\nOptimization log saved to: {opt_log_path}")
print("="*80)
