"""Complete training script for all models with comprehensive results logging."""

from pathlib import Path
import sys
import json
import importlib.util
from datetime import datetime

# Add models directory to path
models_dir = Path(__file__).resolve().parent / "models"
sys.path.insert(0, str(models_dir))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Import pipelines directly
import importlib.util

def load_pipeline(model_name):
    """Load pipeline module from model directory."""
    spec = importlib.util.spec_from_file_location(
        f"{model_name}.pipeline",
        models_dir / model_name / "pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

bpnn_pipeline = load_pipeline("bpnn")
knn_pipeline = load_pipeline("knn")
lr_pipeline = load_pipeline("linear_regression")
rfr_pipeline = load_pipeline("rfr")


def run_model_pipeline(pipeline, model_name: str, results_base_dir: Path) -> dict:
    """Run a complete model pipeline with data loading, training, and evaluation."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} Pipeline")
    print(f"{'='*60}")
    
    try:
        # Load data
        print("Loading agg.data dataset...")
        df = pipeline.load_agg_data()
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Split features and targets
        print("Splitting features and targets...")
        X, y = pipeline.split_features_targets(df)
        print(f"Features: {X.shape}, Targets: {y.shape}")
        
        # Handle NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)
        print(f"After removing NaN: Features: {X.shape}, Targets: {y.shape}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train and evaluate
        print(f"Training {model_name}...")
        metrics = pipeline.train_and_evaluate(X_train, X_test, y_train, y_test, model_name)
        
        # Print metrics
        print(f"\nEvaluation Metrics for {model_name}:")
        print(f"  R² Score: {metrics['R2']:.4f}")
        print(f"  RMSE:     {metrics['RMSE']:.4f}")
        print(f"  MAE:      {metrics['MAE']:.4f}")
        print(f"  MSE:      {metrics['MSE']:.4f}")
        
        # Create model-specific results directory
        model_results_dir = results_base_dir / model_name.replace(" ", "_").lower()
        model_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        print(f"Generating evaluation plots...")
        pipeline.plot_results(metrics, model_results_dir)
        
        # Save metrics to JSON
        metrics_dict = {
            "model": model_name,
            "R2": float(metrics['R2']),
            "RMSE": float(metrics['RMSE']),
            "MAE": float(metrics['MAE']),
            "MSE": float(metrics['MSE']),
            "training_date": datetime.now().isoformat(),
            "train_test_split": "0.8/0.2",
            "random_state": 42,
            "features": ['Concentration (g/mL)', 'Uncoated Layer (nm)', 'Total Thickness (nm)'],
            "target": 'Bonded Thickness (nm)',
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        }
        
        metrics_path = model_results_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        
        return metrics
        
    except Exception as e:
        print(f"Error in {model_name} pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all model pipelines."""
    print("Pitt ML MSE Project - Model Training Pipeline")
    print("=" * 60)
    
    # Create comprehensive results directory
    results_base_dir = Path(__file__).resolve().parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_base_dir}\n")
    
    results = {}
    
    # Run each model
    results["BPNN"] = run_model_pipeline(bpnn_pipeline, "BPNN", results_base_dir)
    results["KNN"] = run_model_pipeline(knn_pipeline, "KNN", results_base_dir)
    results["Linear Regression"] = run_model_pipeline(lr_pipeline, "Linear Regression", results_base_dir)
    results["Random Forest"] = run_model_pipeline(rfr_pipeline, "Random Forest Regression", results_base_dir)
    
    # Create summary report
    print(f"\n{'='*60}")
    print("Summary of Results")
    print(f"{'='*60}")
    
    summary_data = []
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name:25s} | R²: {metrics['R2']:7.4f} | RMSE: {metrics['RMSE']:10.4f} | MAE: {metrics['MAE']:10.4f}")
            summary_data.append({
                "Model": model_name,
                "R2": float(metrics['R2']),
                "RMSE": float(metrics['RMSE']),
                "MAE": float(metrics['MAE']),
                "MSE": float(metrics['MSE']),
            })
    
    # Save summary to CSV and JSON
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = results_base_dir / "summary_metrics.csv"
    summary_json_path = results_base_dir / "summary_metrics.json"
    
    summary_df.to_csv(summary_csv_path, index=False)
    summary_df.to_json(summary_json_path, orient='records', indent=2)
    
    print(f"\nSummary saved to:")
    print(f"  CSV: {summary_csv_path}")
    print(f"  JSON: {summary_json_path}")
    
    # Create a README in results directory
    readme_path = results_base_dir / "README.md"
    readme_content = f"""# Model Training Results

**Training Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

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
"""
    
    for _, row in summary_df.iterrows():
        readme_content += f"| {row['Model']} | {row['R2']:.6f} | {row['RMSE']:.6f} | {row['MAE']:.6f} | {row['MSE']:.6f} |\n"
    
    readme_content += """
## Output Structure

Each model has its own directory containing:
- `{model}_evaluation.png` - Comprehensive evaluation visualization (4 subplots)
- `metrics.json` - Detailed metrics and configuration

See individual model directories for more details.
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  README: {readme_path}")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Results saved to: {results_base_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
