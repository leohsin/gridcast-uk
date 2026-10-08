from pathlib import Path

import pandas as pd

from gridcast_uk.data.neso import (
    process_demand_data,
    save_dataframe,
)


RAW_DATA_DIR = Path("data/raw")

OUTPUT_PATH = Path(
    "data/processed/historical_demand_2021_2026.parquet"
)

YEARS = range(2021, 2027)

processed_years = []

for year in YEARS:
    path = (
        RAW_DATA_DIR
        / f"historic_demand_{year}.parquet"
    )

    print(f"Processing {year}...")

    df = pd.read_parquet(path)

    # Add our own reliable lineage metadata.
    df["SOURCE_YEAR"] = year

    processed_df = process_demand_data(df)

    processed_years.append(processed_df)

    print(
        f"{year}: {len(processed_df):,} rows"
    )

historical_df = pd.concat(
    processed_years,
    ignore_index=True,
    sort=False,
)

historical_df = historical_df.sort_values(
    "settlement_start"
).reset_index(drop=True)

if historical_df["national_demand_mw"].isna().any():
    raise ValueError(
        "Combined dataset contains missing demand values."
    )

duplicate_timestamps = (
    historical_df["settlement_start"]
    .duplicated()
    .sum()
)

if duplicate_timestamps:
    raise ValueError(
        f"Found {duplicate_timestamps} duplicate timestamps."
    )

if not historical_df[
    "settlement_start"
].is_monotonic_increasing:
    raise ValueError(
        "Settlement timestamps are not sorted."
    )

utc_times = (
    historical_df["settlement_start"]
    .dt.tz_convert("UTC")
)

differences = utc_times.diff().dropna()

unexpected_gaps = differences[
    differences != pd.Timedelta(minutes=30)
]

print(
    f"Unexpected timestamp gaps: "
    f"{len(unexpected_gaps):,}"
)

print("\n--- Combined dataset ---")

print(
    f"Rows: {len(historical_df):,}"
)

print(
    "Date range:",
    historical_df["settlement_start"].min(),
    "to",
    historical_df["settlement_start"].max(),
)

print("\nRows by source year:")

print(
    historical_df["source_year"]
    .value_counts()
    .sort_index()
)

print("\nMissing values:")

missing = (
    historical_df.isna()
    .sum()
    .sort_values(ascending=False)
)

print(
    missing[missing > 0]
)

save_dataframe(
    df=historical_df,
    output_path=OUTPUT_PATH,
)

print(
    f"\nSaved dataset to: {OUTPUT_PATH}"
)