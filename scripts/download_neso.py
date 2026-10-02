from pathlib import Path

from gridcast_uk.data.neso import (
    fetch_all_demand_data,
    save_raw_demand_data,
)


OUTPUT_PATH = Path(
    "data/raw/historic_demand_2026.parquet"
)


df = fetch_all_demand_data(
    page_size=1000,
)

print()
print(f"Downloaded {len(df)} rows.")

save_raw_demand_data(
    df=df,
    output_path=OUTPUT_PATH,
)

print(f"Saved raw data to: {OUTPUT_PATH}")