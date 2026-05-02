"""Optimize hyperparameters using training set cross-validation."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def load_holdout_data(split: str = "train") -> pd.DataFrame:
    """Load the holdout validation dataset."""
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    
    if split == "train":
        data_path = data_dir / "train_holdout.csv"
    elif split == "validation":
        data_path = data_dir / "validation_holdout.csv"
    else:
        raise ValueError(f"Invalid split: {split}")
    
    df = pd.read_csv(data_path)
    return df


def get_all_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Get ALL available features and target."""
    all_features = [col for col in df.columns if col != 'Bonded Thickness (nm)']
    target = ['Bonded Thickness (nm)']
    return all_features, target


def optimize_hyperparameters():
    """Find optimal hyperparameters using cross-validation on training set."""
    
    # Load training data
    train_df = load_holdout_data("train")
    feature_cols, target_cols = get_all_features(train_df)
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_cols].values.flatten()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION REPORT")
    print("=" * 70)
    print(f"\nTraining set: {len(X_train)} samples, {len(feature_cols)} features\n")
    
    # 1. BPNN optimization
    print("1. BPNN - Testing hidden layer configurations")
    print("-" * 70)
    bpnn_configs = [
        (64,),
        (128,),
        (256,),
        (100, 50),
        (128, 64),
        (256, 128),
        (128, 64, 32),
        (256, 128, 64),
        (512, 256),
    ]
    
    best_bpnn_cv = -np.inf
    best_bpnn_config = None
    
    for config in bpnn_configs:
        model = MLPRegressor(
            hidden_layer_sizes=config,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            alpha=0.01  # L2 regularization
        )
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        mean_cv = cv_scores.mean()
        
        if mean_cv > best_bpnn_cv:
            best_bpnn_cv = mean_cv
            best_bpnn_config = config
        
        print(f"   Config {str(config):25s}: CV R² = {mean_cv:7.4f} (±{cv_scores.std():.4f})")
    
    print(f"   ✓ Best BPNN: {best_bpnn_config} with CV R² = {best_bpnn_cv:.4f}\n")
    
    # 2. KNN optimization
    print("2. KNN - Testing neighbor counts")
    print("-" * 70)
    knn_configs = [3, 5, 7, 9, 11, 15, 20, 25]
    
    best_knn_cv = -np.inf
    best_knn_config = None
    
    for n_neighbors in knn_configs:
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        mean_cv = cv_scores.mean()
        
        if mean_cv > best_knn_cv:
            best_knn_cv = mean_cv
            best_knn_config = n_neighbors
        
        print(f"   n_neighbors={n_neighbors:2d}: CV R² = {mean_cv:7.4f} (±{cv_scores.std():.4f})")
    
    print(f"   ✓ Best KNN: n_neighbors={best_knn_config} with CV R² = {best_knn_cv:.4f}\n")
    
    # 3. Ridge Regression optimization
    print("3. Ridge Regression - Testing alpha values")
    print("-" * 70)
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    best_ridge_cv = -np.inf
    best_ridge_config = None
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        mean_cv = cv_scores.mean()
        
        if mean_cv > best_ridge_cv:
            best_ridge_cv = mean_cv
            best_ridge_config = alpha
        
        print(f"   alpha={alpha:8.3f}: CV R² = {mean_cv:7.4f} (±{cv_scores.std():.4f})")
    
    print(f"   ✓ Best Ridge: alpha={best_ridge_config} with CV R² = {best_ridge_cv:.4f}\n")
    
    # 4. Random Forest optimization
    print("4. Random Forest - Testing configurations")
    print("-" * 70)
    rf_configs = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 150, "max_depth": 20},
        {"n_estimators": 200, "max_depth": 25},
        {"n_estimators": 100, "max_depth": None},
    ]
    
    best_rf_cv = -np.inf
    best_rf_config = None
    
    for config in rf_configs:
        model = RandomForestRegressor(random_state=42, **config)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        mean_cv = cv_scores.mean()
        
        if mean_cv > best_rf_cv:
            best_rf_cv = mean_cv
            best_rf_config = config
        
        print(f"   n_est={config['n_estimators']:3d}, depth={str(config['max_depth']):>4s}: CV R² = {mean_cv:7.4f} (±{cv_scores.std():.4f})")
    
    print(f"   ✓ Best RF: {best_rf_config} with CV R² = {best_rf_cv:.4f}\n")
    
    # 5. Gradient Boosting
    print("5. Gradient Boosting - Testing configurations")
    print("-" * 70)
    gb_configs = [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5},
        {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 6},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 7},
    ]
    
    best_gb_cv = -np.inf
    best_gb_config = None
    
    for config in gb_configs:
        model = GradientBoostingRegressor(random_state=42, subsample=0.8, **config)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        mean_cv = cv_scores.mean()
        
        if mean_cv > best_gb_cv:
            best_gb_cv = mean_cv
            best_gb_config = config
        
        print(f"   n_est={config['n_estimators']:3d}, lr={config['learning_rate']:.2f}, depth={config['max_depth']}: CV R² = {mean_cv:7.4f} (±{cv_scores.std():.4f})")
    
    print(f"   ✓ Best GB: {best_gb_config} with CV R² = {best_gb_cv:.4f}\n")
    
    # Summary
    print("=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"\nBest models by CV R² score:")
    print(f"  1. Gradient Boosting:  {best_gb_cv:.4f}")
    print(f"  2. Random Forest:      {best_rf_cv:.4f}")
    print(f"  3. BPNN:               {best_bpnn_cv:.4f}")
    print(f"  4. Ridge:              {best_ridge_cv:.4f}")
    print(f"  5. KNN:                {best_knn_cv:.4f}")
    
    print("\nOptimal configurations:")
    print(f"  BPNN hidden layers:    {best_bpnn_config}")
    print(f"  KNN neighbors:         {best_knn_config}")
    print(f"  Ridge alpha:           {best_ridge_config}")
    print(f"  Random Forest:         {best_rf_config}")
    print(f"  Gradient Boosting:     {best_gb_config}")
    
    return {
        "bpnn": {"config": best_bpnn_config, "cv_score": best_bpnn_cv},
        "knn": {"config": best_knn_config, "cv_score": best_knn_cv},
        "ridge": {"config": best_ridge_config, "cv_score": best_ridge_cv},
        "rf": {"config": best_rf_config, "cv_score": best_rf_cv},
        "gb": {"config": best_gb_config, "cv_score": best_gb_cv},
    }


if __name__ == "__main__":
    optimize_hyperparameters()
