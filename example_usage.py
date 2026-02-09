"""Example usage of model pipelines with agg.data dataset and evaluation metrics."""

from pathlib import Path
import sys

# Add models directory to path
models_dir = Path(__file__).resolve().parent / "models"

# Import pipeline modules
sys.path.insert(0, str(models_dir / "bpnn"))
sys.path.insert(0, str(models_dir / "knn"))
sys.path.insert(0, str(models_dir / "linear_regression"))
sys.path.insert(0, str(models_dir / "rfr"))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import pipelines
import bpnn.pipeline as bpnn_pipeline
import knn.pipeline as knn_pipeline
import linear_regression.pipeline as lr_pipeline
import rfr.pipeline as rfr_pipeline


def run_model_pipeline(pipeline, model_name: str):
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
        
        # Generate plots
        print(f"Generating evaluation plots...")
        pipeline.plot_results(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"Error in {model_name} pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all model pipelines."""
    print("Pitt ML MSE Project - Model Pipeline Example")
    print("=" * 60)
    
    results = {}
    
    # Run each model
    results["BPNN"] = run_model_pipeline(bpnn_pipeline, "BPNN")
    results["KNN"] = run_model_pipeline(knn_pipeline, "KNN")
    results["Linear Regression"] = run_model_pipeline(lr_pipeline, "Linear Regression")
    results["Random Forest"] = run_model_pipeline(rfr_pipeline, "Random Forest Regression")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary of Results")
    print(f"{'='*60}")
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name:25s} | R²: {metrics['R2']:7.4f} | RMSE: {metrics['RMSE']:10.4f}")


if __name__ == "__main__":
    main()
