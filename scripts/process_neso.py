from pathlib import Path

import pandas as pd

from gridcast_uk.data.neso import (
    process_demand_data,
    save_dataframe,
)


RAW_PATH = Path(
    "data/raw/historic_demand_2026.parquet"
)

PROCESSED_PATH = Path(
    "data/processed/demand_2026.parquet"
)


raw_df = pd.read_parquet(
    RAW_PATH
)

print(
    f"Loaded {len(raw_df):,} raw rows."
)

processed_df = process_demand_data(
    raw_df
)

save_dataframe(
    df=processed_df,
    output_path=PROCESSED_PATH,
)

print(
    f"Saved {len(processed_df):,} processed rows."
)

print(
    f"Output: {PROCESSED_PATH}"
)