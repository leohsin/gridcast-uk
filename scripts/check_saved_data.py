from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/raw/historic_demand_2026.parquet"
)

df = pd.read_parquet(DATA_PATH)

print(df.head())

print()
print("Shape:")
print(df.shape)

print()
print("Columns:")
print(df.columns.tolist())