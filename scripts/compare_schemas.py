from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")

YEARS = range(2021, 2027)

schemas = {}


for year in YEARS:
    path = DATA_DIR / f"historic_demand_{year}.parquet"

    df = pd.read_parquet(path)

    schemas[year] = set(df.columns)

    print(f"\n{year}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(df.columns.tolist())

common_columns = set.intersection(
    *schemas.values()
)

all_columns = set.union(
    *schemas.values()
)


print("\n--- Columns present in ALL years ---")

for column in sorted(common_columns):
    print(column)


print("\n--- Columns not present in every year ---")

for column in sorted(all_columns - common_columns):
    years_present = [
        year
        for year, columns in schemas.items()
        if column in columns
    ]

    print(
        f"{column}: {years_present}"
    )