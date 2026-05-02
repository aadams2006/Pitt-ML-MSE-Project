"""Prepare holdout validation data with 2000 training and 1000 validation samples."""

from pathlib import Path

import pandas as pd


def prepare_holdout_data():
    """
    Split the data into training (first 2000) and validation (last 1000) sets.
    
    The splits are deterministic based on index order, not random.
    """
    # Load data
    data_path = Path(__file__).resolve().parents[1] / "Data" / "synthetic_data_improved.csv"
    df = pd.read_csv(data_path)
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Verify we have enough data
    assert len(df) >= 3000, f"Expected at least 3000 rows, got {len(df)}"
    
    # Split: first 2000 for training, last 1000 for validation
    train_data = df.iloc[:2000].reset_index(drop=True)
    validation_data = df.iloc[-1000:].reset_index(drop=True)
    
    # Save to separate files
    output_dir = Path(__file__).resolve().parents[1] / "Data"
    
    train_path = output_dir / "train_holdout.csv"
    validation_path = output_dir / "validation_holdout.csv"
    
    train_data.to_csv(train_path, index=False)
    validation_data.to_csv(validation_path, index=False)
    
    print(f"Training data: {len(train_data)} samples saved to {train_path}")
    print(f"Validation data: {len(validation_data)} samples saved to {validation_path}")
    
    return train_path, validation_path


if __name__ == "__main__":
    prepare_holdout_data()
