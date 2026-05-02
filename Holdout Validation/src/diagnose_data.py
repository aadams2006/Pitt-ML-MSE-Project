"""Diagnostic analysis of holdout validation data."""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def diagnose():
    """Run diagnostic checks on the data."""
    
    # Load data
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    train_df = pd.read_csv(data_dir / "train_holdout.csv")
    val_df = pd.read_csv(data_dir / "validation_holdout.csv")
    
    print("=" * 70)
    print("DATA DIAGNOSTIC REPORT")
    print("=" * 70)
    
    # 1. Data shape and basic stats
    print("\n1. DATA SHAPES")
    print(f"   Training: {train_df.shape}")
    print(f"   Validation: {val_df.shape}")
    
    # 2. Column names
    print("\n2. AVAILABLE COLUMNS")
    for col in train_df.columns:
        print(f"   - {col}")
    
    # 3. Feature correlations with target
    print("\n3. CORRELATION WITH TARGET (Bonded Thickness)")
    target_col = "Bonded Thickness (nm)"
    for col in train_df.columns:
        if col != target_col:
            corr = train_df[col].corr(train_df[target_col])
            print(f"   {col:40s}: {corr:7.4f}")
    
    # 4. Distribution comparison
    print("\n4. DISTRIBUTION COMPARISON (Train vs Validation)")
    print(f"   {'Column':<40} {'Train Mean':>12} {'Val Mean':>12} {'Diff':>12}")
    print("   " + "-" * 76)
    for col in train_df.columns:
        train_mean = train_df[col].mean()
        val_mean = val_df[col].mean()
        diff = val_mean - train_mean
        print(f"   {col:<40} {train_mean:12.4f} {val_mean:12.4f} {diff:12.4f}")
    
    # 5. Current features vs all available features
    print("\n5. CURRENT MODEL CONFIGURATION")
    current_features = ['Concentration (g/mL)', 'Uncoated Layer (nm)', 'Total Thickness (nm)']
    all_features = [col for col in train_df.columns if col != target_col]
    print(f"   Current features: {len(current_features)}")
    for f in current_features:
        print(f"     - {f}")
    print(f"\n   All available features: {len(all_features)}")
    for f in all_features:
        print(f"     - {f}")
    
    # 6. Basic statistics on target variable
    print("\n6. TARGET VARIABLE STATISTICS")
    print(f"   {'':40} {'Train':>12} {'Validation':>12}")
    print("   " + "-" * 64)
    print(f"   {'Mean':<40} {train_df[target_col].mean():12.4f} {val_df[target_col].mean():12.4f}")
    print(f"   {'Std':<40} {train_df[target_col].std():12.4f} {val_df[target_col].std():12.4f}")
    print(f"   {'Min':<40} {train_df[target_col].min():12.4f} {val_df[target_col].min():12.4f}")
    print(f"   {'Max':<40} {train_df[target_col].max():12.4f} {val_df[target_col].max():12.4f}")
    
    # 7. Check for missing values
    print("\n7. MISSING VALUES")
    print(f"   Training: {train_df.isnull().sum().sum()}")
    print(f"   Validation: {val_df.isnull().sum().sum()}")
    
    # 8. Check for infinite values
    print("\n8. INFINITE VALUES")
    train_inf = np.isinf(train_df.values).sum()
    val_inf = np.isinf(val_df.values).sum()
    print(f"   Training: {train_inf}")
    print(f"   Validation: {val_inf}")
    
    # 9. Variance in current features
    print("\n9. VARIANCE IN CURRENT FEATURES (scaled)")
    X_train = train_df[current_features]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(f"   {'Feature':<40} {'Variance':>12}")
    for i, feat in enumerate(current_features):
        var = np.var(X_train_scaled[:, i])
        print(f"   {feat:<40} {var:12.6f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    diagnose()
