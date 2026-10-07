from pathlib import Path

import pandas as pd

from gridcast_uk.data.neso import save_dataframe
from gridcast_uk.features.demand_features import (
    build_demand_features,
)


INPUT_PATH = Path(
    "data/processed/"
    "historical_demand_2021_2026.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "demand_features_2021_2026.parquet"
)


df = pd.read_parquet(INPUT_PATH)

feature_df = build_demand_features(df)


print("--- Feature dataset ---")
print(f"Rows: {len(feature_df):,}")
print(f"Columns: {len(feature_df.columns)}")

feature_columns = [
    "settlement_start",
    "national_demand_mw",
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "demand_lag_1d_mw",
    "demand_lag_2d_mw",
    "demand_lag_7d_mw",
    "demand_change_1d_mw",
]

print()
print(feature_df[feature_columns].head(10))

print()
print("--- Missing feature values ---")

print(
    feature_df[
        [
            "demand_lag_1d_mw",
            "demand_lag_2d_mw",
            "demand_lag_7d_mw",
            "demand_change_1d_mw",
        ]
    ]
    .isna()
    .sum()
)


save_dataframe(
    feature_df,
    OUTPUT_PATH,
)

print()
print(f"Saved to: {OUTPUT_PATH}")