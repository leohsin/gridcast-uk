from datetime import date
from pathlib import Path

from gridcast_uk.data.neso import (
    save_dataframe,
)
from gridcast_uk.data.weather import (
    fetch_gb_weather_forecasts,
)


OUTPUT_PATH = Path(
    "data/raw/"
    "weather_forecasts_2026_by_location.parquet"
)


weather_df = fetch_gb_weather_forecasts(
    start_date=date(2026, 1, 1),
    end_date=date(2026, 9, 11),
)


print(
    "\n--- 2026 regional weather ---"
)

print(
    f"Rows: {len(weather_df):,}"
)

print(
    f"Locations: "
    f"{weather_df['location'].nunique()}"
)

print(
    f"Range: "
    f"{weather_df['time'].min()} "
    f"to "
    f"{weather_df['time'].max()}"
)


weather_columns = [
    "temperature_2m_previous_day1",
    "shortwave_radiation_instant_previous_day1",
    "cloud_cover_previous_day1",
    "wind_speed_10m_previous_day1",
]


print(
    "\n--- Missing values by location ---"
)

missing = (
    weather_df
    .groupby("location")[weather_columns]
    .apply(
        lambda x: x.isna().sum()
    )
)

print(
    missing.to_string()
)


save_dataframe(
    weather_df,
    OUTPUT_PATH,
)


print(
    f"\nSaved to: {OUTPUT_PATH}"
)