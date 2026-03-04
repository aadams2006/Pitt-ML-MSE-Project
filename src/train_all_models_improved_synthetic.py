"""Training script using improved synthetic data that preserves correlations."""

import sys
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import importlib.util

# Add models directory to path
models_dir = Path(__file__).resolve().parent.parent / "models"

def load_pipeline_synthetic(model_name):
    """Load synthetic pipeline module from model directory."""
    spec = importlib.util.spec_from_file_location(
        f"{model_name}.pipeline_synthetic",
        models_dir / model_name / "pipeline_synthetic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load pipelines
bpnn_pipeline = load_pipeline_synthetic("bpnn")
knn_pipeline = load_pipeline_synthetic("knn")
lr_pipeline = load_pipeline_synthetic("linear_regression")
rfr_pipeline = load_pipeline_synthetic("rfr")


def main():
    """Run all model pipelines with improved synthetic data."""
    print("="*60)
    print("Training with IMPROVED Synthetic Data (Correlations Preserved)")
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
    results_base_dir = Path(__file__).resolve().parent.parent / "results" / f"improved_synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        print(f"Training {model_name}")
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
                "train_test_split": "0.8/0.2",
                "random_state": 42,
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "data_source": "synthetic_data_improved.csv (correlation-preserving)"
            }
            
            with open(model_results_dir / "metrics.json", 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            results[model_name] = metrics
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Improved Synthetic Data Results")
    print(f"{'='*60}\n")
    
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
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_base_dir / "summary_metrics.csv", index=False)
    summary_df.to_json(results_base_dir / "summary_metrics.json", orient='records', indent=2)
    
    print(f"\nResults saved to: {results_base_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
