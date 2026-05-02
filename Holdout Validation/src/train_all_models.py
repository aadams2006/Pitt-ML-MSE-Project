"""Train all models on holdout validation data and collect results."""

import json
import importlib.util
from datetime import datetime
from pathlib import Path


def load_module(name, path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Get base directory
base_dir = Path(__file__).resolve().parent.parent

# Load modules
models_dir = base_dir / "models"
src_dir = base_dir / "src"

pipeline_bpnn = load_module("pipeline_bpnn", models_dir / "pipeline_bpnn.py")
pipeline_knn = load_module("pipeline_knn", models_dir / "pipeline_knn.py")
pipeline_lr = load_module("pipeline_linear_regression", models_dir / "pipeline_linear_regression.py")
pipeline_rf = load_module("pipeline_random_forest", models_dir / "pipeline_random_forest.py")
prepare_module = load_module("prepare_holdout_data", src_dir / "prepare_holdout_data.py")

# Extract functions
train_bpnn = pipeline_bpnn.train_and_evaluate
plot_bpnn = pipeline_bpnn.plot_results
train_knn = pipeline_knn.train_and_evaluate
plot_knn = pipeline_knn.plot_results
train_lr = pipeline_lr.train_and_evaluate
plot_lr = pipeline_lr.plot_results
train_rf = pipeline_rf.train_and_evaluate
plot_rf = pipeline_rf.plot_results
prepare_holdout_data = prepare_module.prepare_holdout_data


def main():
    """Train all models and save results."""
    
    # Prepare data splits
    print("Preparing data splits...")
    prepare_holdout_data()
    
    # Initialize results storage
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"holdout_validation_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    all_metrics = []
    
    # Train BPNN
    print("\nTraining BPNN...")
    try:
        metrics_bpnn = train_bpnn()
        all_metrics.append(metrics_bpnn)
        plot_bpnn(metrics_bpnn, run_dir)
        print(f"  R²: {metrics_bpnn['R2']:.4f}, RMSE: {metrics_bpnn['RMSE']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Train KNN
    print("\nTraining KNN...")
    try:
        metrics_knn = train_knn()
        all_metrics.append(metrics_knn)
        plot_knn(metrics_knn, run_dir)
        print(f"  R²: {metrics_knn['R2']:.4f}, RMSE: {metrics_knn['RMSE']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Train Linear Regression
    print("\nTraining Linear Regression...")
    try:
        metrics_lr = train_lr()
        all_metrics.append(metrics_lr)
        plot_lr(metrics_lr, run_dir)
        print(f"  R²: {metrics_lr['R2']:.4f}, RMSE: {metrics_lr['RMSE']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Train Random Forest
    print("\nTraining Random Forest Regression...")
    try:
        metrics_rf = train_rf()
        all_metrics.append(metrics_rf)
        plot_rf(metrics_rf, run_dir)
        print(f"  R²: {metrics_rf['R2']:.4f}, RMSE: {metrics_rf['RMSE']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Save summary metrics
    summary = {}
    for metrics in all_metrics:
        model_name = metrics["model"]
        summary[model_name] = {
            "MSE": float(metrics["MSE"]),
            "RMSE": float(metrics["RMSE"]),
            "MAE": float(metrics["MAE"]),
            "R2": float(metrics["R2"]),
        }
    
    summary_path = run_dir / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {run_dir}")
    print(f"Summary metrics: {summary_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("HOLDOUT VALIDATION RESULTS SUMMARY")
    print("="*60)
    print(f"Training samples: 2000")
    print(f"Validation samples: 1000")
    print()
    for model_name, metrics in summary.items():
        print(f"{model_name}:")
        print(f"  R² Score: {metrics['R2']:.6f}")
        print(f"  RMSE:     {metrics['RMSE']:.6f}")
        print(f"  MAE:      {metrics['MAE']:.6f}")
        print(f"  MSE:      {metrics['MSE']:.6f}")
        print()


if __name__ == "__main__":
    main()
