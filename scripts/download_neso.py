from pathlib import Path

from gridcast_uk.data.neso import (
    fetch_all_demand_data,
    save_raw_demand_data,
    validate_raw_demand_data,
    print_demand_data_summary,
)


OUTPUT_PATH = Path(
    "data/raw/historic_demand_2026.parquet"
)


df = fetch_all_demand_data(
    page_size=1000,
)

print()
print(f"Downloaded {len(df)} rows.")

print("Validating raw data...")

validate_raw_demand_data(df)

print("Validation passed.")

save_raw_demand_data(
    df=df,
    output_path=OUTPUT_PATH,
)

print(f"Saved raw data to: {OUTPUT_PATH}")

print_demand_data_summary(df)