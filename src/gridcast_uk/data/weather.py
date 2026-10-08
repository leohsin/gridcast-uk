from datetime import date

import httpx
import pandas as pd


PREVIOUS_RUNS_URL = (
    "https://previous-runs-api.open-meteo.com/v1/forecast"
)


def fetch_previous_day_weather_forecast(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    hourly_variables = [
        "temperature_2m_previous_day1",
        "shortwave_radiation_instant_previous_day1",
        "cloud_cover_previous_day1",
        "wind_speed_10m_previous_day1",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(hourly_variables),
        "timezone": "UTC",
    }

    response = httpx.get(
        PREVIOUS_RUNS_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            f"Weather API returned no hourly data: {data}"
        )

    weather_df = pd.DataFrame(
        data["hourly"]
    )

    weather_df["time"] = pd.to_datetime(
        weather_df["time"],
        utc=True,
    )

    return weather_df

WEATHER_LOCATIONS = {
    "london": (51.5074, -0.1278),
    "birmingham": (52.4862, -1.8904),
    "manchester": (53.4808, -2.2426),
    "leeds": (53.8008, -1.5491),
    "newcastle": (54.9783, -1.6178),
    "cardiff": (51.4816, -3.1791),
    "glasgow": (55.8642, -4.2518),
    "edinburgh": (55.9533, -3.1883),
}

def fetch_gb_weather_forecasts(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    frames = []

    for location, (
        latitude,
        longitude,
    ) in WEATHER_LOCATIONS.items():

        print(
            f"Downloading weather: {location}"
        )

        location_df = (
            fetch_previous_day_weather_forecast(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
            )
        )

        location_df["location"] = location

        frames.append(location_df)

    return pd.concat(
        frames,
        ignore_index=True,
    )

def aggregate_gb_weather(
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    aggregated = (
        weather_df
        .groupby("time")
        .agg(
            forecast_temperature_c=(
                "temperature_2m_previous_day1",
                "mean",
            ),
            forecast_shortwave_radiation=(
                "shortwave_radiation_instant_previous_day1",
                "mean",
            ),
            forecast_cloud_cover_pct=(
                "cloud_cover_previous_day1",
                "mean",
            ),
            forecast_wind_speed=(
                "wind_speed_10m_previous_day1",
                "mean",
            ),
        )
        .reset_index()
    )

    return aggregated

def prepare_half_hour_weather(
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    weather_df = weather_df.copy()

    weather_columns = [
        "forecast_temperature_c",
        "forecast_shortwave_radiation",
        "forecast_cloud_cover_pct",
        "forecast_wind_speed",
    ]

    weather_df["time"] = pd.to_datetime(
        weather_df["time"],
        utc=True,
    )

    weather_df = weather_df.sort_values(
        "time"
    )

    # Solar radiation is the last variable to
    # become continuously available.
    weather_df = weather_df[
        weather_df["time"]
        >= pd.Timestamp(
            "2024-03-07 00:00:00",
            tz="UTC",
        )
    ].copy()

    if weather_df[weather_columns].isna().any().any():
        missing = (
            weather_df[weather_columns]
            .isna()
            .sum()
        )

        raise ValueError(
            "Weather data still contains missing "
            f"values after cutoff:\n{missing}"
        )

    half_hour_weather = (
        weather_df
        .set_index("time")[weather_columns]
        .resample("30min")
        .interpolate(method="time")
        .reset_index()
    )

    return half_hour_weather