from gridcast_uk.data.neso import (
    add_settlement_timestamp,
    fetch_demand_records,
    records_to_dataframe,
)


data = fetch_demand_records(limit=100)

df = records_to_dataframe(data)

df = add_settlement_timestamp(df)

print(
    df[
        [
            "SETTLEMENT_DATE",
            "SETTLEMENT_PERIOD",
            "SETTLEMENT_START",
            "ND",
        ]
    ].head(10)
)

print("\n--- Data types ---")
print(
    df[
        [
            "SETTLEMENT_DATE",
            "SETTLEMENT_START",
            "ND",
        ]
    ].dtypes
)