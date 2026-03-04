"""
Improved synthetic data generator that preserves feature-target correlations.
Uses bootstrap resampling with correlated noise to maintain the relationship structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.linalg import cholesky


def generate_synthetic_data_preserving_correlations(num_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data that preserves feature-target correlations from original data.
    
    Uses bootstrap resampling combined with correlated noise to maintain the relationship
    structure while creating new samples.
    
    Args:
        num_rows (int): Number of synthetic samples to generate.
        seed (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Synthetic data preserving original correlations.
    """
    np.random.seed(seed)
    
    # Load original data
    data_path = Path(__file__).resolve().parents[1] / "data" / "agg.data.xlsx"
    original_data = pd.read_excel(data_path)
    
    print(f"Original data shape: {original_data.shape}")
    print(f"Original correlations with target:")
    print(original_data.corr()['Bonded Thickness (nm)'])
    
    # Calculate correlation matrix from original data
    correlation_matrix = original_data.corr().values
    
    # Use Cholesky decomposition to generate correlated noise
    try:
        L = cholesky(correlation_matrix, lower=True)
    except np.linalg.LinAlgError:
        print("Warning: Correlation matrix not positive definite. Using SVD approximation.")
        U, s, _ = np.linalg.svd(correlation_matrix)
        L = U @ np.diag(np.sqrt(np.abs(s)))
    
    # Get original statistics
    means = original_data.mean().values
    stds = original_data.std().values
    
    # Generate synthetic data using bootstrap + correlated noise
    synthetic_samples = []
    
    # Bootstrap from original data and add correlated noise
    for _ in range(num_rows // 10):  # 10% pure bootstrap
        idx = np.random.randint(0, len(original_data))
        synthetic_samples.append(original_data.iloc[idx].values)
    
    # 90% with added correlated noise
    for _ in range(num_rows - len(synthetic_samples)):
        # Start with bootstrap sample
        idx = np.random.randint(0, len(original_data))
        base_sample = original_data.iloc[idx].values.copy()
        
        # Generate correlated noise
        z = np.random.normal(0, 1, len(original_data.columns))
        noise = L @ z
        
        # Scale noise proportionally to std of each feature
        noise = noise * (stds / stds.max()) * 0.3  # 30% of std as noise level
        
        # Add noise to base sample
        noisy_sample = base_sample + noise
        
        # Ensure realistic bounds based on original data ranges
        for j in range(len(noisy_sample)):
            min_val = original_data.iloc[:, j].min()
            max_val = original_data.iloc[:, j].max()
            # Allow 10% beyond original range for extrapolation
            range_val = max_val - min_val
            noisy_sample[j] = np.clip(
                noisy_sample[j],
                min_val - 0.1 * range_val,
                max_val + 0.1 * range_val
            )
        
        synthetic_samples.append(noisy_sample)
    
    # Create dataframe
    synthetic_data = pd.DataFrame(
        synthetic_samples,
        columns=original_data.columns
    )
    
    # Verify correlations are preserved
    print(f"\nGenerated {len(synthetic_data)} synthetic samples")
    print(f"Synthetic data correlations with target:")
    print(synthetic_data.corr()['Bonded Thickness (nm)'])
    
    # Calculate correlation preservation quality
    original_target_corr = original_data.corr()['Bonded Thickness (nm)'].drop('Bonded Thickness (nm)')
    synthetic_target_corr = synthetic_data.corr()['Bonded Thickness (nm)'].drop('Bonded Thickness (nm)')
    
    print(f"\n=== Correlation Preservation Analysis ===")
    for col in original_data.columns[:-1]:
        orig_corr = original_target_corr[col]
        synth_corr = synthetic_target_corr[col]
        pct_preservation = (1 - abs(orig_corr - synth_corr) / (abs(orig_corr) + 0.001)) * 100
        print(f"{col:25s}: Original={orig_corr:7.4f}, Synthetic={synth_corr:7.4f}, Preservation={pct_preservation:6.1f}%")
    
    return synthetic_data


if __name__ == "__main__":
    synthetic_data = generate_synthetic_data_preserving_correlations(num_rows=1000)
    
    # Save the synthetic data
    output_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_data_improved.csv"
    synthetic_data.to_csv(output_path, index=False)
    print(f"\nSynthetic data saved to {output_path}")
    
    # Also print basic statistics
    print(f"\n=== Data Statistics ===")
    print(synthetic_data.describe())
