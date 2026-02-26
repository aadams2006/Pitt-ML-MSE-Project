import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Load the real data
data = pd.read_excel('data/agg.data.xlsx')

# Create metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)

# Create a CTGAN synthesizer
synthesizer = CTGANSynthesizer(metadata)

# Fit the synthesizer to the data
synthesizer.fit(data)

# Generate 1000 synthetic samples
synthetic_data = synthesizer.sample(num_rows=1000)

# Save the synthetic data
synthetic_data.to_csv('data/synthetic_data.csv', index=False)

print("Synthetic data generated and saved to data/synthetic_data.csv")
