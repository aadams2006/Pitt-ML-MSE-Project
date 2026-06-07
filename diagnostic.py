import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path("analytical model comparision/ethyl acetate")
df = pd.read_csv(base_dir / "ethyl acetate+pdms.csv")
print("Sample of data showing concentration vs bonded thickness:")
print(df[['Concentration (g/mL)', 'Bonded Thickness (nm)']].head(15))
print("\nGrouped by concentration:")
grouped = df.groupby('Concentration (g/mL)')['Bonded Thickness (nm)'].agg(['count', 'mean', 'std', 'min', 'max'])
print(grouped)
print("\nCorrelation with concentration:")
print(f"Pearson correlation: {df['Concentration (g/mL)'].corr(df['Bonded Thickness (nm)']):.4f}")

