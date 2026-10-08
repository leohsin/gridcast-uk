from pathlib import Path

import pandas as pd

from gridcast_uk.data.neso import save_dataframe


WEATHER_PATH = Path(
    "data/raw/"
    "weather_forecasts_2026_by_location.parquet"
)

DEMAND_PATH = Path(
    "data/processed/"
    "demand_features_2021_2026.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "demand_features_with_regional_weather_2026.parquet"
)


WEATHER_COLUMN_NAMES = {
    "temperature_2m_previous_day1": "temperature_c",
    "shortwave_radiation_instant_previous_day1": "shortwave_radiation",
    "cloud_cover_previous_day1": "cloud_cover_pct",
    "wind_speed_10m_previous_day1": "wind_speed",
}


# --------------------------------------------------
# 1. Load hourly regional weather
# --------------------------------------------------

weather_df = pd.read_parquet(
    WEATHER_PATH
)

weather_df["time"] = pd.to_datetime(
    weather_df["time"],
    utc=True,
)


print("--- Raw 2026 regional weather ---")

print(
    f"Rows: {len(weather_df):,}"
)

print(
    f"Locations: "
    f"{weather_df['location'].nunique()}"
)


# --------------------------------------------------
# 2. Reshape location-level weather
#
# Before:
#
# time | location | temperature | ...
#
# After:
#
# time | london_temperature_c | ...
# --------------------------------------------------

wide_weather = weather_df.pivot(
    index="time",
    columns="location",
    values=list(
        WEATHER_COLUMN_NAMES.keys()
    ),
)


wide_weather.columns = [
    (
        f"{location}_"
        f"{WEATHER_COLUMN_NAMES[variable]}"
    )
    for variable, location
    in wide_weather.columns
]


wide_weather = (
    wide_weather
    .sort_index()
)


print(
    f"Hourly timestamps: "
    f"{len(wide_weather):,}"
)

print(
    f"Regional weather features: "
    f"{len(wide_weather.columns)}"
)


# --------------------------------------------------
# 3. Convert hourly forecasts to half-hourly
# --------------------------------------------------

half_hour_weather = (
    wide_weather
    .resample("30min")
    .interpolate(method="time")
    .reset_index()
)


print(
    "\n--- Half-hour weather ---"
)

print(
    f"Rows: {len(half_hour_weather):,}"
)

print(
    f"Range: "
    f"{half_hour_weather['time'].min()} "
    f"to "
    f"{half_hour_weather['time'].max()}"
)


# --------------------------------------------------
# 4. Check weather missingness
# --------------------------------------------------

weather_feature_columns = [
    column
    for column in half_hour_weather.columns
    if column != "time"
]


missing_weather = (
    half_hour_weather[
        weather_feature_columns
    ]
    .isna()
    .sum()
)


print(
    "\n--- Missing half-hour weather values ---"
)

print(
    missing_weather[
        missing_weather > 0
    ]
)


if missing_weather.sum() > 0:
    raise ValueError(
        "Half-hour regional weather "
        "contains missing values."
    )


# --------------------------------------------------
# 5. Load demand features
# --------------------------------------------------

demand_df = pd.read_parquet(
    DEMAND_PATH
)


# Keep only the untouched 2026 test period.
demand_df = demand_df[
    demand_df["source_year"] == 2026
].copy()


# Convert UK-local settlement timestamps to UTC
# so they match the weather timestamps.
demand_df["time"] = (
    demand_df["settlement_start"]
    .dt.tz_convert("UTC")
)


print(
    "\n--- 2026 demand data ---"
)

print(
    f"Rows: {len(demand_df):,}"
)

print(
    f"Range UTC: "
    f"{demand_df['time'].min()} "
    f"to "
    f"{demand_df['time'].max()}"
)


# --------------------------------------------------
# 6. Merge demand and weather
# --------------------------------------------------

merged_df = demand_df.merge(
    half_hour_weather,
    on="time",
    how="left",
    validate="one_to_one",
)


merged_missing_weather = (
    merged_df[
        weather_feature_columns
    ]
    .isna()
    .sum()
)


print(
    "\n--- Merged 2026 test dataset ---"
)

print(
    f"Rows: {len(merged_df):,}"
)

print(
    f"Columns: {len(merged_df.columns):,}"
)

print(
    "\nMissing weather values after merge:"
)

print(
    merged_missing_weather[
        merged_missing_weather > 0
    ]
)


if merged_missing_weather.sum() > 0:
    raise ValueError(
        "2026 test dataset contains "
        "missing weather values."
    )


# --------------------------------------------------
# 7. Check target
# --------------------------------------------------

print(
    f"\nMissing target values: "
    f"{merged_df['national_demand_mw'].isna().sum():,}"
)


# --------------------------------------------------
# 8. Save
# --------------------------------------------------

save_dataframe(
    merged_df,
    OUTPUT_PATH,
)


print(
    f"\nSaved to: {OUTPUT_PATH}"
)