from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/raw/weather_forecasts_2024_2025.parquet"
)

WEATHER_COLUMNS = [
    "forecast_temperature_c",
    "forecast_shortwave_radiation",
    "forecast_cloud_cover_pct",
    "forecast_wind_speed",
]


df = pd.read_parquet(DATA_PATH)


print("--- First valid timestamp ---")

for column in WEATHER_COLUMNS:
    first_valid = df.loc[
        df[column].notna(),
        "time",
    ].min()

    print(
        f"{column}: {first_valid}"
    )


print("\n--- Last valid timestamp ---")

for column in WEATHER_COLUMNS:
    last_valid = df.loc[
        df[column].notna(),
        "time",
    ].max()

    print(
        f"{column}: {last_valid}"
    )


df["year_month"] = (
    df["time"]
    .dt.tz_localize(None)
    .dt.to_period("M")
)


missing_by_month = (
    df
    .groupby("year_month")[WEATHER_COLUMNS]
    .apply(
        lambda x: x.isna().sum()
    )
)


print("\n--- Missing values by month ---")

print(
    missing_by_month[
        missing_by_month.sum(axis=1) > 0
    ].to_string()
)


def print_missing_blocks(
    data: pd.DataFrame,
    column: str,
) -> None:
    missing_rows = data.loc[
        data[column].isna(),
        ["time"],
    ].copy()

    if missing_rows.empty:
        print(
            f"\n{column}: no missing values"
        )
        return

    missing_rows["new_block"] = (
        missing_rows["time"].diff()
        != pd.Timedelta(hours=1)
    )

    missing_rows["block"] = (
        missing_rows["new_block"].cumsum()
    )

    blocks = (
        missing_rows
        .groupby("block")
        .agg(
            start=("time", "min"),
            end=("time", "max"),
            hours=("time", "size"),
        )
    )

    print(
        f"\n--- Missing blocks: {column} ---"
    )

    print(
        blocks.to_string(
            index=False
        )
    )


for column in WEATHER_COLUMNS:
    print_missing_blocks(
        df,
        column,
    )