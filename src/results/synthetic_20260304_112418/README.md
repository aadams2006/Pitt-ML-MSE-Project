# Model Training Results (Synthetic Data)

**Training Date:** 2026-03-04 11:24:23

## Dataset Configuration
- **Data File:** synthetic_data.csv
- **Features:** 
  - Concentration (g/mL)
  - Uncoated Layer (nm)
  - Total Thickness (nm)
- **Target:** Bonded Thickness (nm)
- **Train/Test Split:** 80/20
- **Random State:** 42

## Models Trained
1. BPNN (Backpropagation Neural Network)
2. KNN (K-Nearest Neighbors)
3. Linear Regression
4. Random Forest Regression

## Results

| Model | R² Score | RMSE | MAE | MSE |
|-------|----------|------|-----|-----|
| BPNN | -0.085849 | 0.115590 | 0.096266 | 0.013361 |
| KNN | -0.117925 | 0.117285 | 0.095990 | 0.013756 |
| Linear Regression | -0.002976 | 0.111092 | 0.095336 | 0.012341 |
| Random Forest | -0.131553 | 0.117998 | 0.095756 | 0.013924 |

## Output Structure

Each model has its own directory containing:
- `{model}_evaluation.png` - Comprehensive evaluation visualization (4 subplots)
- `metrics.json` - Detailed metrics and configuration

See individual model directories for more details.
