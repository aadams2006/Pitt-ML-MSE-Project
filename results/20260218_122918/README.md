# Model Training Results

**Training Date:** 2026-02-18 12:29:21

## Dataset Configuration
- **Data File:** agg.data.xlsx
- **Features:** 
  - Concentration (g/mL)
  - Uncoated Layer (nm)
  - Total Thickness (nm)
- **Target:** Bonded Thickness (nm)
- **Total Samples:** 58
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
| BPNN | 0.111385 | 0.068248 | 0.062426 | 0.004658 |
| KNN | 0.645342 | 0.043116 | 0.031500 | 0.001859 |
| Linear Regression | 0.429546 | 0.054682 | 0.046016 | 0.002990 |
| Random Forest | 0.898470 | 0.023069 | 0.019919 | 0.000532 |

## Output Structure

Each model has its own directory containing:
- `{model}_evaluation.png` - Comprehensive evaluation visualization (4 subplots)
- `metrics.json` - Detailed metrics and configuration

See individual model directories for more details.
