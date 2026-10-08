from pathlib import Path

import pandas as pd

from gridcast_uk.data.neso import save_dataframe


WEATHER_PATH = Path(
    "data/raw/"
    "weather_forecasts_2024_2025_by_location.parquet"
)

DEMAND_PATH = Path(
    "data/processed/"
    "demand_features_2021_2026.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "demand_features_with_regional_weather_2024_2025.parquet"
)


WEATHER_COLUMN_NAMES = {
    "temperature_2m_previous_day1": "temperature_c",
    "shortwave_radiation_instant_previous_day1": "shortwave_radiation",
    "cloud_cover_previous_day1": "cloud_cover_pct",
    "wind_speed_10m_previous_day1": "wind_speed",
}


# --------------------------------------------------
# 1. Load regional hourly weather
# --------------------------------------------------

weather_df = pd.read_parquet(
    WEATHER_PATH
)

weather_df["time"] = pd.to_datetime(
    weather_df["time"],
    utc=True,
)


# Use the same safe archive cutoff as before.
weather_df = weather_df[
    weather_df["time"]
    >= pd.Timestamp(
        "2024-03-07 00:00:00",
        tz="UTC",
    )
].copy()


# --------------------------------------------------
# 2. Reshape from long format to wide format
#
# Before:
#
# time | location | temperature | radiation ...
#
# After:
#
# time | london_temperature | glasgow_temperature ...
# --------------------------------------------------

wide_weather = weather_df.pivot(
    index="time",
    columns="location",
    values=list(WEATHER_COLUMN_NAMES.keys()),
)


# Flatten pandas MultiIndex column names.
#
# Example:
#
# ("temperature_2m_previous_day1", "london")
#
# becomes:
#
# london_temperature_c
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
    "--- Hourly regional weather ---"
)

print(
    f"Rows: {len(wide_weather):,}"
)

print(
    f"Weather features: "
    f"{len(wide_weather.columns)}"
)


# --------------------------------------------------
# 3. Convert hourly weather to half-hourly
#
# 12:00 known forecast
# 12:30 interpolated
# 13:00 known forecast
# --------------------------------------------------

half_hour_weather = (
    wide_weather
    .resample("30min")
    .interpolate(method="time")
    .reset_index()
)


print(
    "\n--- Half-hour regional weather ---"
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
# 4. Create national-average weather features
#
# We include these in the SAME dataset so the later
# experiment can compare national averages against
# regional features on exactly the same observations.
# --------------------------------------------------

locations = sorted(
    weather_df["location"]
    .unique()
    .tolist()
)


for weather_name in WEATHER_COLUMN_NAMES.values():

    regional_columns = [
        f"{location}_{weather_name}"
        for location in locations
    ]

    half_hour_weather[
        f"forecast_{weather_name}"
    ] = (
        half_hour_weather[
            regional_columns
        ].mean(axis=1)
    )


# --------------------------------------------------
# 5. Check weather missingness
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
    "\n--- Missing weather values ---"
)

print(
    missing_weather[
        missing_weather > 0
    ]
)

if missing_weather.sum() > 0:
    raise ValueError(
        "Regional weather contains missing values."
    )


# --------------------------------------------------
# 6. Load demand features
# --------------------------------------------------

demand_df = pd.read_parquet(
    DEMAND_PATH
)


# Convert GB settlement timestamp to UTC.
demand_df["time"] = (
    demand_df["settlement_start"]
    .dt.tz_convert("UTC")
)


# Keep only the weather-covered period.
demand_df = demand_df[
    demand_df["time"].between(
        half_hour_weather["time"].min(),
        half_hour_weather["time"].max(),
    )
].copy()


# --------------------------------------------------
# 7. Merge demand and regional weather
# --------------------------------------------------

merged_df = demand_df.merge(
    half_hour_weather,
    on="time",
    how="left",
    validate="one_to_one",
)


print(
    "\n--- Merged regional dataset ---"
)

print(
    f"Rows: {len(merged_df):,}"
)

print(
    f"Columns: {len(merged_df.columns):,}"
)

print(
    f"Range: "
    f"{merged_df['time'].min()} "
    f"to "
    f"{merged_df['time'].max()}"
)


merged_missing_weather = (
    merged_df[
        weather_feature_columns
    ]
    .isna()
    .sum()
    .sum()
)


print(
    f"Missing weather values after merge: "
    f"{merged_missing_weather:,}"
)


if merged_missing_weather > 0:
    raise ValueError(
        "Merged dataset contains missing "
        "regional weather values."
    )


# --------------------------------------------------
# 8. Save processed dataset
# --------------------------------------------------

save_dataframe(
    merged_df,
    OUTPUT_PATH,
)


print(
    f"\nSaved to: {OUTPUT_PATH}"
)