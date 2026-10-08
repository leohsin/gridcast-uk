from datetime import date

from gridcast_uk.data.weather import (
    fetch_previous_day_weather_forecast,
)


# Approximate central London coordinates
LONDON_LATITUDE = 51.5074
LONDON_LONGITUDE = -0.1278


weather_df = (
    fetch_previous_day_weather_forecast(
        latitude=LONDON_LATITUDE,
        longitude=LONDON_LONGITUDE,
        start_date=date(2025, 3, 14),
        end_date=date(2025, 3, 20),
    )
)


print("--- Weather forecast data ---")

print(
    f"Rows: {len(weather_df):,}"
)

print(
    f"Date range: "
    f"{weather_df['time'].min()} "
    f"to "
    f"{weather_df['time'].max()}"
)

print()

print(weather_df.head(10))

print()

print("--- Missing values ---")

print(
    weather_df
    .isna()
    .sum()
)

march_18 = weather_df[
    weather_df["time"].dt.date
    == date(2025, 3, 18)
]

print(
    "\n--- March 18 weather forecast ---"
)

print(
    march_18.to_string(
        index=False,
    )
)