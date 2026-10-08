from datetime import date
from pathlib import Path

from gridcast_uk.data.neso import (
    save_dataframe,
)
from gridcast_uk.data.weather import (
    aggregate_gb_weather,
    fetch_gb_weather_forecasts,
)


RAW_LOCATION_PATH = Path(
    "data/raw/"
    "weather_forecasts_2024_2025_by_location.parquet"
)

AGGREGATED_PATH = Path(
    "data/raw/"
    "weather_forecasts_2024_2025.parquet"
)


# -----------------------------------
# Download forecasts for all locations
# -----------------------------------

weather_df = fetch_gb_weather_forecasts(
    start_date=date(2024, 1, 1),
    end_date=date(2025, 12, 31),
)


print(
    "\n--- Location-level weather data ---"
)

print(
    f"Rows: {len(weather_df):,}"
)

print(
    f"Locations: "
    f"{weather_df['location'].nunique()}"
)


save_dataframe(
    weather_df,
    RAW_LOCATION_PATH,
)

print(
    f"Saved location data to: "
    f"{RAW_LOCATION_PATH}"
)


# -----------------------------------
# Build national averages
# -----------------------------------

gb_weather_df = aggregate_gb_weather(
    weather_df
)


print(
    "\n--- GB aggregated weather data ---"
)

print(
    f"Rows: {len(gb_weather_df):,}"
)

print(
    f"Range: "
    f"{gb_weather_df['time'].min()} "
    f"to "
    f"{gb_weather_df['time'].max()}"
)

print(
    "\nMissing values:"
)

print(
    gb_weather_df.isna().sum()
)


save_dataframe(
    gb_weather_df,
    AGGREGATED_PATH,
)

print(
    f"\nSaved aggregated data to: "
    f"{AGGREGATED_PATH}"
)