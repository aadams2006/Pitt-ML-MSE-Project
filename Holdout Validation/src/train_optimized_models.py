"""Train all optimized models on holdout validation data and collect results."""

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

# Load optimized modules
models_dir = base_dir / "models"
src_dir = base_dir / "src"

pipeline_bpnn_opt = load_module("pipeline_bpnn_optimized", models_dir / "pipeline_bpnn_optimized.py")
pipeline_rf_opt = load_module("pipeline_rf_optimized", models_dir / "pipeline_rf_optimized.py")
pipeline_gb_opt = load_module("pipeline_gb_optimized", models_dir / "pipeline_gb_optimized.py")
prepare_module = load_module("prepare_holdout_data", src_dir / "prepare_holdout_data.py")

# Extract functions
train_bpnn_opt = pipeline_bpnn_opt.train_and_evaluate
plot_bpnn_opt = pipeline_bpnn_opt.plot_results
train_rf_opt = pipeline_rf_opt.train_and_evaluate
plot_rf_opt = pipeline_rf_opt.plot_results
train_gb_opt = pipeline_gb_opt.train_and_evaluate
plot_gb_opt = pipeline_gb_opt.plot_results
prepare_holdout_data = prepare_module.prepare_holdout_data


def main():
    """Train all optimized models and save results."""
    
    # Prepare data splits
    print("Preparing data splits...")
    prepare_holdout_data()
    
    # Initialize results storage
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"holdout_optimized_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    all_metrics = []
    
    # Train BPNN Optimized
    print("\nTraining BPNN (Optimized) with 7 features...")
    try:
        metrics_bpnn = train_bpnn_opt()
        all_metrics.append(metrics_bpnn)
        plot_bpnn_opt(metrics_bpnn, run_dir)
        print(f"  R²: {metrics_bpnn['R2']:.4f}, RMSE: {metrics_bpnn['RMSE']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Train Random Forest Optimized
    print("\nTraining Random Forest (Optimized) with 7 features...")
    try:
        metrics_rf = train_rf_opt()
        all_metrics.append(metrics_rf)
        plot_rf_opt(metrics_rf, run_dir)
        print(f"  R²: {metrics_rf['R2']:.4f}, RMSE: {metrics_rf['RMSE']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Train Gradient Boosting Optimized
    print("\nTraining Gradient Boosting (Optimized) with 7 features...")
    try:
        metrics_gb = train_gb_opt()
        all_metrics.append(metrics_gb)
        plot_gb_opt(metrics_gb, run_dir)
        print(f"  R²: {metrics_gb['R2']:.4f}, RMSE: {metrics_gb['RMSE']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
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
    print("\n" + "="*70)
    print("OPTIMIZED HOLDOUT VALIDATION RESULTS")
    print("="*70)
    print(f"Training samples: 2000")
    print(f"Validation samples: 1000")
    print(f"Features: All 7 available features (not just 3)")
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
