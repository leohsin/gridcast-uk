from pathlib import Path

from gridcast_uk.data.neso import (
    fetch_all_demand_data,
    save_dataframe,
    validate_raw_demand_data,
)


YEARS = range(2021, 2027)

RAW_DATA_DIR = Path("data/raw")


for year in YEARS:
    output_path = (
        RAW_DATA_DIR
        / f"historic_demand_{year}.parquet"
    )

    if output_path.exists():
        print(
            f"{year}: file already exists, skipping."
        )
        continue

    print()
    print(f"Downloading {year}...")

    df = fetch_all_demand_data(
        year=year,
        page_size=1000,
    )

    print(f"\nColumns for {year}:")
    print(df.columns.tolist())

    validate_raw_demand_data(df)

    save_dataframe(
        df=df,
        output_path=output_path,
    )

    print(
        f"Saved {len(df):,} rows to {output_path}"
    )