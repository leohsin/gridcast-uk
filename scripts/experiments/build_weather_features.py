from pathlib import Path

import pandas as pd

from gridcast_uk.data.neso import (
    save_dataframe,
)
from gridcast_uk.data.weather import (
    prepare_half_hour_weather,
)


DEMAND_PATH = Path(
    "data/processed/"
    "demand_features_2021_2026.parquet"
)

WEATHER_PATH = Path(
    "data/raw/"
    "weather_forecasts_2024_2025.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "demand_features_with_weather_2024_2025.parquet"
)


# -------------------------
# Load weather
# -------------------------

weather_df = pd.read_parquet(
    WEATHER_PATH
)

weather_df = prepare_half_hour_weather(
    weather_df
)


print("--- Half-hour weather ---")

print(
    f"Rows: {len(weather_df):,}"
)

print(
    f"Range: "
    f"{weather_df['time'].min()} "
    f"to "
    f"{weather_df['time'].max()}"
)


# -------------------------
# Load demand features
# -------------------------

demand_df = pd.read_parquet(
    DEMAND_PATH
)

# settlement_start is UK local time with
# timezone information.
#
# Convert it to UTC so it can be matched
# exactly to the weather timestamps.
demand_df["time"] = (
    demand_df["settlement_start"]
    .dt.tz_convert("UTC")
)


# Keep only periods covered by weather data.
demand_df = demand_df[
    demand_df["time"].between(
        weather_df["time"].min(),
        weather_df["time"].max(),
    )
].copy()


# -------------------------
# Merge
# -------------------------

merged_df = demand_df.merge(
    weather_df,
    on="time",
    how="left",
    validate="one_to_one",
)


weather_columns = [
    "forecast_temperature_c",
    "forecast_shortwave_radiation",
    "forecast_cloud_cover_pct",
    "forecast_wind_speed",
]


print("\n--- Merged demand + weather ---")

print(
    f"Rows: {len(merged_df):,}"
)

print(
    f"Range: "
    f"{merged_df['time'].min()} "
    f"to "
    f"{merged_df['time'].max()}"
)

print("\nMissing weather values:")

print(
    merged_df[
        weather_columns
    ]
    .isna()
    .sum()
)


save_dataframe(
    merged_df,
    OUTPUT_PATH,
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)