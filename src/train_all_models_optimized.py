"""Training script using OPTIMIZED hyperparameters on improved synthetic data."""

import sys
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import importlib.util

# Add models directory to path
models_dir = Path(__file__).resolve().parent.parent / "models"

def load_pipeline_optimized(model_name):
    """Load optimized pipeline module from model directory."""
    spec = importlib.util.spec_from_file_location(
        f"{model_name}.pipeline_optimized",
        models_dir / model_name / "pipeline_optimized.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load optimized pipelines
bpnn_pipeline = load_pipeline_optimized("bpnn")
knn_pipeline = load_pipeline_optimized("knn")
lr_pipeline = load_pipeline_optimized("linear_regression")
rfr_pipeline = load_pipeline_optimized("rfr")


def main():
    """Run all model pipelines with improved synthetic data using OPTIMIZED hyperparameters."""
    print("="*60)
    print("Training with IMPROVED Synthetic Data (OPTIMIZED Hyperparameters)")
    print("="*60)
    
    # Load improved synthetic data
    data_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_data_improved.csv"
    df = pd.read_csv(data_path)
    
    print(f"\nLoaded improved synthetic data: {df.shape}")
    print(f"\nTarget correlations:")
    target_corr = df.corr()["Bonded Thickness (nm)"]
    for col in df.columns[:-1]:
        print(f"  {col:25s}: {target_corr[col]:7.4f}")
    
    # Create results directory
    results_base_dir = Path(__file__).resolve().parent.parent / "results" / f"optimized_hyperparams_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_base_dir}\n")
    
    results = {}
    models = [
        ("BPNN", bpnn_pipeline),
        ("KNN", knn_pipeline),
        ("Linear Regression", lr_pipeline),
        ("Random Forest Regression", rfr_pipeline)
    ]
    
    # Train each model
    for model_name, pipeline in models:
        print(f"\n{'='*60}")
        print(f"Training {model_name} (OPTIMIZED)")
        print(f"{'='*60}")
        
        try:
            # Split and prepare data
            X, y = pipeline.split_features_targets(df)
            valid_idx = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
            X = X[valid_idx].reset_index(drop=True)
            y = y[valid_idx].reset_index(drop=True)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Train and evaluate
            metrics = pipeline.train_and_evaluate(X_train, X_test, y_train, y_test, model_name)
            
            print(f"\nEvaluation Metrics:")
            print(f"  R² Score:  {metrics['R2']:7.4f}")
            print(f"  RMSE:      {metrics['RMSE']:10.4f}")
            print(f"  MAE:       {metrics['MAE']:10.4f}")
            print(f"  MSE:       {metrics['MSE']:10.4f}")
            
            # Save results
            model_results_dir = results_base_dir / model_name.replace(" ", "_").lower()
            model_results_dir.mkdir(parents=True, exist_ok=True)
            
            pipeline.plot_results(metrics, model_results_dir)
            
            metrics_dict = {
                "model": model_name,
                "R2": float(metrics['R2']),
                "RMSE": float(metrics['RMSE']),
                "MAE": float(metrics['MAE']),
                "MSE": float(metrics['MSE']),
                "training_date": datetime.now().isoformat(),
                "hyperparameter_optimization": "OPTIMIZED"
            }
            
            # Save individual metrics
            metrics_file = model_results_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            print(f"Metrics saved to {metrics_file}")
            
            results[model_name] = metrics_dict
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary metrics
    summary_file = results_base_dir / "summary_metrics.json"
    summary_list = list(results.values())
    with open(summary_file, 'w') as f:
        json.dump(summary_list, f, indent=2)
    print(f"\n\nSummary metrics saved to {summary_file}")
    
    # Also save as CSV for easy comparison
    summary_csv = results_base_dir / "summary_metrics.csv"
    summary_df = pd.DataFrame([
        {
            "Model": r["model"],
            "R2": r["R2"],
            "RMSE": r["RMSE"],
            "MAE": r["MAE"],
            "MSE": r["MSE"]
        }
        for r in summary_list
    ])
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to {summary_csv}")
    
    # Create README with hyperparameter information
    readme_path = results_base_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# Optimized Hyperparameter Training Results\n\n")
        f.write("This directory contains results from training models with OPTIMIZED hyperparameters.\n\n")
        f.write("## Optimization Details\n\n")
        f.write("Optimization was performed using GridSearchCV (BPNN, KNN) and RandomizedSearchCV (RFR).\n")
        f.write("See `src/optimization_results/` for detailed optimization logs.\n\n")
        f.write("## Models Trained\n\n")
        f.write("1. **BPNN** - Backpropagation Neural Networks\n")
        f.write("2. **KNN** - K-Nearest Neighbors\n")
        f.write("3. **Linear Regression** - Linear Model\n")
        f.write("4. **Random Forest Regression** - Ensemble Model\n\n")
        f.write("## Results Summary\n\n")
        for model_name, metrics in results.items():
            f.write(f"### {model_name}\n")
            f.write(f"- R² Score: {metrics['R2']:.6f}\n")
            f.write(f"- RMSE: {metrics['RMSE']:.6f}\n")
            f.write(f"- MAE: {metrics['MAE']:.6f}\n")
            f.write(f"- MSE: {metrics['MSE']:.6f}\n\n")
    print(f"README saved to {readme_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
