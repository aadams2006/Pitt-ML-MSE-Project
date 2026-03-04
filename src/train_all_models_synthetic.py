"""Complete training script for all models with comprehensive results logging using synthetic data."""

from pathlib import Path
import sys
import json
import importlib.util
from datetime import datetime

# Add models directory to path
models_dir = Path(__file__).resolve().parent.parent / "models"
sys.path.insert(0, str(models_dir))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Import pipelines directly
import importlib.util

def load_pipeline_synthetic(model_name):
    """Load synthetic pipeline module from model directory."""
    spec = importlib.util.spec_from_file_location(
        f"{model_name}.pipeline_synthetic",
        models_dir / model_name / "pipeline_synthetic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

bpnn_pipeline_synthetic = load_pipeline_synthetic("bpnn")
knn_pipeline_synthetic = load_pipeline_synthetic("knn")
lr_pipeline_synthetic = load_pipeline_synthetic("linear_regression")
rfr_pipeline_synthetic = load_pipeline_synthetic("rfr")


def run_model_pipeline_synthetic(pipeline, model_name: str, results_base_dir: Path) -> dict:
    """Run a complete model pipeline with synthetic data loading, training, and evaluation."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} Pipeline with Synthetic Data")
    print(f"{'='*60}")
    
    try:
        # Load data
        print("Loading synthetic dataset...")
        df = pipeline.load_synthetic_data()
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
        print(f"Train set: {X_train.shape}, Test set: {y_test.shape}")
        
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
            "data_source": "synthetic_data.csv"
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
    """Run all model pipelines with synthetic data."""
    print("Pitt ML MSE Project - Model Training Pipeline with Synthetic Data")
    print("=" * 60)
    
    # Create comprehensive results directory
    results_base_dir = Path(__file__).resolve().parent / "results" / f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_base_dir}\n")
    
    results = {}
    
    # Run each model
    results["BPNN"] = run_model_pipeline_synthetic(bpnn_pipeline_synthetic, "BPNN", results_base_dir)
    results["KNN"] = run_model_pipeline_synthetic(knn_pipeline_synthetic, "KNN", results_base_dir)
    results["Linear Regression"] = run_model_pipeline_synthetic(lr_pipeline_synthetic, "Linear Regression", results_base_dir)
    results["Random Forest"] = run_model_pipeline_synthetic(rfr_pipeline_synthetic, "Random Forest Regression", results_base_dir)
    
    # Create summary report
    print(f"\n{'='*60}")
    print("Summary of Results (Synthetic Data)")
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
    readme_content = f"""# Model Training Results (Synthetic Data)\n\n**Training Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n## Dataset Configuration\n- **Data File:** synthetic_data.csv\n- **Features:** \n  - Concentration (g/mL)\n  - Uncoated Layer (nm)\n  - Total Thickness (nm)\n- **Target:** Bonded Thickness (nm)\n- **Train/Test Split:** 80/20\n- **Random State:** 42\n\n## Models Trained\n1. BPNN (Backpropagation Neural Network)\n2. KNN (K-Nearest Neighbors)\n3. Linear Regression\n4. Random Forest Regression\n\n## Results\n\n| Model | R² Score | RMSE | MAE | MSE |\n|-------|----------|------|-----|-----|\n"""
    
    for _, row in summary_df.iterrows():
        readme_content += f"| {row['Model']} | {row['R2']:.6f} | {row['RMSE']:.6f} | {row['MAE']:.6f} | {row['MSE']:.6f} |\n"
    
    readme_content += """\n## Output Structure\n\nEach model has its own directory containing:\n- `{model}_evaluation.png` - Comprehensive evaluation visualization (4 subplots)\n- `metrics.json` - Detailed metrics and configuration\n\nSee individual model directories for more details.\n"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  README: {readme_path}")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Results saved to: {results_base_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()