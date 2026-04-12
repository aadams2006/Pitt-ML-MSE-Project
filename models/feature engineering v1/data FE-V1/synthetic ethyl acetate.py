"""Generate a synthetic ethyl acetate dataset via bootstrap + Cholesky noise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _cholesky_cov(cov: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Compute a stable Cholesky factor with small diagonal jitter if needed."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov_jittered = cov + np.eye(cov.shape[0]) * jitter
        return np.linalg.cholesky(cov_jittered)


def generate_synthetic(
    source_df: pd.DataFrame,
    n_samples: int = 1000,
    noise_scale: float = 0.05,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap rows and add correlated noise from Cholesky decomposition."""
    rng = np.random.default_rng(random_seed)
    numeric_df = source_df.select_dtypes(include=[np.number]).copy()

    # Bootstrap base samples
    base = numeric_df.sample(n=n_samples, replace=True, random_state=random_seed).to_numpy()

    # Correlated Gaussian noise using Cholesky of covariance
    cov = np.cov(numeric_df.to_numpy().T, bias=False)
    chol = _cholesky_cov(cov)
    z = rng.standard_normal(size=base.shape)
    noise = (z @ chol.T) * noise_scale

    synthetic = base + noise

    # Clip to observed min/max to keep values plausible
    mins = numeric_df.min().to_numpy()
    maxs = numeric_df.max().to_numpy()
    synthetic = np.clip(synthetic, mins, maxs)

    synthetic_df = pd.DataFrame(synthetic, columns=numeric_df.columns)
    return synthetic_df


def correlation_similarity(original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
    """Correlation between the upper triangles of the correlation matrices."""
    corr_orig = original_df.corr(numeric_only=True)
    corr_syn = synthetic_df.corr(numeric_only=True)
    idx = np.triu_indices_from(corr_orig, k=1)
    v_orig = corr_orig.to_numpy()[idx]
    v_syn = corr_syn.to_numpy()[idx]
    if v_orig.size == 0:
        return float("nan")
    return float(np.corrcoef(v_orig, v_syn)[0, 1])


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "ethyl acetate+pdms.csv"
    output_path = base_dir / "synthetic_ethyl_acetate.csv"

    df = pd.read_csv(input_path)
    synthetic_df = generate_synthetic(df, n_samples=1000, noise_scale=0.05, random_seed=42)
    synthetic_df.to_csv(output_path, index=False)

    corr_coef = correlation_similarity(df, synthetic_df)
    print(f"Synthetic dataset saved to: {output_path}")
    print(f"Correlation coefficient (corr-matrix similarity): {corr_coef:.4f}")


if __name__ == "__main__":
    main()
