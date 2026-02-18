import pandas as pd
from pathlib import Path

data_path = Path("data/agg.data.xlsx")
df = pd.read_excel(data_path)

print("Column names:")
print(df.columns.tolist())
print("\nDataframe shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
