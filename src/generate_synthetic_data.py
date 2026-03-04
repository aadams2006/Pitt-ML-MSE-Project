import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path

def generate_and_return_synthetic_data(num_rows: int = 1000) -> pd.DataFrame:
    """
    Generates synthetic data using CTGAN and returns it as a pandas DataFrame.
    
    Args:
        num_rows (int): The number of synthetic samples to generate.
        
    Returns:
        pd.DataFrame: The generated synthetic data.
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "agg.data.xlsx"
    
    # Load the real data
    data = pd.read_excel(data_path)

    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)

    # Create a CTGAN synthesizer
    synthesizer = CTGANSynthesizer(metadata)

    # Fit the synthesizer to the data
    synthesizer.fit(data)

    # Generate synthetic samples
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    return synthetic_data

if __name__ == "__main__":
    synthetic_data = generate_and_return_synthetic_data()
    # Save the synthetic data
    output_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_data.csv"
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data generated and saved to {output_path}")
