# Holdout Validation Setup

This folder contains the holdout validation framework for evaluating machine learning models with a fixed train/validation split.

## ✓ Optimization Complete

**Best Model**: Random Forest (Optimized)  
**Performance**: R² = 0.328, RMSE = 0.106  
**Improvement**: +1.55 R² from original baseline (-1.22 → 0.33)

See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) for detailed analysis.

## Data Split

- **Training Set**: First 2,000 datapoints
- **Validation Set**: Last 1,000 datapoints
- **Total**: 3,000 datapoints from `synthetic_data_improved.csv`

## Features and Target

### Original Configuration (3 Features)
**Features:**
- Concentration (g/mL)
- Uncoated Layer (nm)
- Total Thickness (nm)

⚠️ **Note**: Poor generalization due to limited features

### Optimized Configuration (7 Features) ✓ RECOMMENDED
**Features:**
- Polarity (XLogP3)
- Viscosity (cP)
- Boiling Point (K)
- Surface Tension (mN/m)
- Concentration (g/mL)
- Uncoated Layer (nm)
- Total Thickness (nm)

**Target:**
- Bonded Thickness (nm)

## Structure

### Directory Layout
```
Holdout Validation/
├── Data/
│   ├── synthetic_data_improved.csv     (Original data)
│   ├── train_holdout.csv               (Generated: first 2000 rows)
│   └── validation_holdout.csv          (Generated: last 1000 rows)
├── models/
│   ├── pipeline_bpnn.py                (BPNN model)
│   ├── pipeline_knn.py                 (KNN model)
│   ├── pipeline_linear_regression.py   (Linear Regression model)
│   ├── pipeline_random_forest.py       (Random Forest model)
│   └── results/                        (Generated: model outputs)
├── src/
│   ├── prepare_holdout_data.py         (Data preparation script)
│   └── train_all_models.py             (Master training script)
└── results/                            (Generated: training results)
```

## Usage

### Quick Start - Run Optimized Models
```bash
python src/compare_models.py
```
This runs all models and shows detailed comparison.

### Compare Original vs Optimized
```bash
python src/compare_models.py
```
Outputs comparison table and analysis.

### Train Only Optimized Models
```bash
python src/train_optimized_models.py
```

### Train Specific Optimized Model
```bash
# Best overall model
python models/pipeline_rf_optimized.py

# Alternative ultra-optimized variant
python models/pipeline_rf_ultra_optimized.py
```

### Diagnostic Analysis
```bash
python src/diagnose_data.py
```
Shows distribution analysis and feature correlations.

## 1. Prepare Data (Automatic)
The data split is automatically created when running the training script:

```bash
python src/train_all_models.py
```

Or manually prepare the splits:

```bash
python src/prepare_holdout_data.py
```

### 2. Train All Models
Run the master training script to train all four models:

```bash
python src/train_all_models.py
```

This will:
- Prepare the data splits if not already done
- Train all four models (BPNN, KNN, Linear Regression, Random Forest)
- Generate evaluation plots for each model
- Save summary metrics as JSON
- Print a summary of all results

### 3. Train Individual Models
To train a specific model:

```bash
# BPNN
python models/pipeline_bpnn.py

# KNN
python models/pipeline_knn.py

# Linear Regression
python models/pipeline_linear_regression.py

# Random Forest
python models/pipeline_random_forest.py
```

## Output

Results are saved in the `results/` folder with a timestamp:
```
results/
└── holdout_validation_YYYYMMDD_HHMMSS/
    ├── summary_metrics.json
    ├── BPNN_evaluation.png
    ├── KNN_evaluation.png
    ├── Linear_Regression_evaluation.png
    └── Random_Forest_Regression_evaluation.png
```

Each evaluation PNG contains:
- Predictions vs Actual scatter plot
- Residual plot
- Performance metrics (R², RMSE, MAE, MSE)
- Residual distribution histogram

## Model Details

### ✓ Recommended: Random Forest (Optimized)
**Best holdout validation performance (R² = 0.328)**
- Number of estimators: 200
- Max depth: 20
- Min samples split: 5
- Min samples leaf: 2
- Feature subsampling: sqrt(n_features)
- Features: All 7
- **Holdout R²: 0.328** ✓

### Alternative: Random Forest (Ultra-Optimized)
- Number of estimators: 500 (more conservative)
- Max depth: 10 (shallower)
- Features: All 7
- **Holdout R²: 0.317** (similar performance, slightly different regularization)

### Original Models (Not Recommended)
#### BPNN
- Hidden layers: (100, 50)
- Max iterations: 1000
- Solver: adam
- Features: 3
- **Holdout R²: -331.96** ✗ (extremely poor)

#### KNN
- Number of neighbors: 5
- Features: 3
- **Holdout R²: -2.23** ✗ (poor)

#### Random Forest (Original - 3 Features)
- Number of estimators: 100
- Features: 3
- **Holdout R²: -1.22** ✗ (negative)

## Performance Metrics

Each model generates the following metrics:
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Squared Error (lower is better)
