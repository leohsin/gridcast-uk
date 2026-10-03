from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/demand_2026.parquet"
)


df = pd.read_parquet(DATA_PATH)


print("\n--- First rows ---")
print(
    df[
        [
            "settlement_start",
            "settlement_period",
            "nd",
        ]
    ].head(10)
)


print("\n--- Last rows ---")
print(
    df[
        [
            "settlement_start",
            "settlement_period",
            "nd",
        ]
    ].tail(10)
)


print("\n--- Shape ---")
print(df.shape)


print("\n--- Data types ---")
print(
    df[
        [
            "settlement_date",
            "settlement_start",
            "settlement_period",
            "nd",
        ]
    ].dtypes
)