from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/raw/"
    "weather_forecasts_2024_2025_by_location.parquet"
)


WEATHER_COLUMNS = [
    "temperature_2m_previous_day1",
    "shortwave_radiation_instant_previous_day1",
    "cloud_cover_previous_day1",
    "wind_speed_10m_previous_day1",
]


df = pd.read_parquet(
    DATA_PATH
)


df["time"] = pd.to_datetime(
    df["time"],
    utc=True,
)


# Use the same safe cutoff as the
# national weather experiment.
df = df[
    df["time"]
    >= pd.Timestamp(
        "2024-03-07 00:00:00",
        tz="UTC",
    )
].copy()


print(
    "--- Rows by location ---"
)

print(
    df.groupby("location")
    .size()
    .sort_index()
)


print(
    "\n--- Missing values by location ---"
)

missing_by_location = (
    df
    .groupby("location")[WEATHER_COLUMNS]
    .apply(
        lambda x: x.isna().sum()
    )
)

print(
    missing_by_location.to_string()
)