from typing import Any
from pathlib import Path

import httpx
import pandas as pd

API_URL = (
    "https://" +
    "api.neso.energy" +
    "/api/3/action/datastore_search"
)

HISTORIC_DEMAND_RESOURCE_IDS = {
    2021: "18c69c42-f20d-46f0-84e9-e279045befc6",
    2022: "bb44a1b5-75b1-4db2-8491-257f23385006",
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
    2025: "b2bde559-3455-4021-b179-dfe60c0337b0",
    2026: "8a4a771c-3929-4e56-93ad-cdf13219dea5",
}


def fetch_demand_records(
    year: int,
    limit: int = 1000,
    offset: int = 0,
) -> dict[str, Any]:
    if year not in HISTORIC_DEMAND_RESOURCE_IDS:
        raise ValueError(
            f"No NESO resource configured for year {year}."
        )

    resource_id = HISTORIC_DEMAND_RESOURCE_IDS[year]

    params = {
        "resource_id": resource_id,
        "limit": limit,
        "offset": offset,
    }

    response = httpx.get(
        API_URL,
        params=params,
        timeout=30.0,
    )

    response.raise_for_status()

    data = response.json()

    if not data["success"]:
        raise RuntimeError(
            f"NESO API request for {year} was unsuccessful."
        )

    return data

def records_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    records = data["result"]["records"]

    df = pd.DataFrame(records)

    df["SETTLEMENT_DATE"] = pd.to_datetime(
        df["SETTLEMENT_DATE"]
    )

    return df

def add_settlement_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    timestamps = []

    for settlement_date, group in df.groupby("SETTLEMENT_DATE"):
        start = settlement_date.tz_localize("Europe/London")

        next_day = settlement_date + pd.Timedelta(days=1)
        end = next_day.tz_localize("Europe/London")

        settlement_times = pd.date_range(
            start=start,
            end=end,
            freq="30min",
            inclusive="left",
        )

        mapping = {
            period: timestamp
            for period, timestamp in enumerate(
                settlement_times,
                start=1,
            )
        }

        group = group.copy()

        group["SETTLEMENT_START"] = (
            group["SETTLEMENT_PERIOD"].map(mapping)
        )

        timestamps.append(group)

    result = pd.concat(timestamps)

    result = result.sort_values(
        ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]
    ).reset_index(drop=True)

    return result

def fetch_all_demand_records(
    year: int,
    page_size: int = 1000,
) -> list[dict[str, Any]]:
    all_records = []
    offset = 0

    while True:
        data = fetch_demand_records(
            year=year,
            limit=page_size,
            offset=offset,
        )

        result = data["result"]

        records = result["records"]
        total = result["total"]

        all_records.extend(records)

        print(
            f"{year}: downloaded "
            f"{len(all_records):,} "
            f"of {total:,} records..."
        )

        if len(all_records) >= total:
            break

        offset += page_size

    return all_records

def fetch_all_demand_data(
    year: int,
    page_size: int = 1000,
) -> pd.DataFrame:
    records = fetch_all_demand_records(
        year=year,
        page_size=page_size,
    )

    df = pd.DataFrame(records)

    df["SOURCE_YEAR"] = year

    return df

def save_raw_demand_data(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

def validate_raw_demand_data(df: pd.DataFrame) -> None:
    required_columns = {
        "SETTLEMENT_DATE",
        "SETTLEMENT_PERIOD",
        "ND",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["ND"].isna().any():
        raise ValueError("ND contains missing values.")

    if (df["ND"] < 0).any():
        raise ValueError("ND contains negative values.")

    if df["SETTLEMENT_DATE"].isna().any():
        raise ValueError(
            "SETTLEMENT_DATE contains missing values."
        )

    if df["SETTLEMENT_PERIOD"].isna().any():
        raise ValueError(
            "SETTLEMENT_PERIOD contains missing values."
        )

    if not df["SETTLEMENT_PERIOD"].between(1, 50).all():
        raise ValueError(
            "SETTLEMENT_PERIOD contains values outside 1-50."
        )

    duplicate_mask = df.duplicated(
        subset=[
            "SETTLEMENT_DATE",
            "SETTLEMENT_PERIOD",
        ]
    )

    if duplicate_mask.any():
        duplicate_count = duplicate_mask.sum()

        raise ValueError(
            f"Found {duplicate_count} duplicate settlement periods."
        )

    if "FORECAST_ACTUAL_INDICATOR" in df.columns:
      valid_indicators = {"A", "F"}

      observed_indicators = set(
          df["FORECAST_ACTUAL_INDICATOR"]
          .dropna()
          .unique()
      )

      invalid_indicators = (
          observed_indicators - valid_indicators
      )

      if invalid_indicators:
          raise ValueError(
              "Unexpected FORECAST_ACTUAL_INDICATOR "
              f"values: {invalid_indicators}"
          )

def print_demand_data_summary(
    df: pd.DataFrame,
) -> None:
    print("\n--- Demand data summary ---")

    print(f"Rows: {len(df)}")

    print(
        "Date range:"
        f" {df['SETTLEMENT_DATE'].min()}"
        f" to {df['SETTLEMENT_DATE'].max()}"
    )

    print(
        f"Minimum ND: {df['ND'].min():,} MW"
    )

    print(
        f"Maximum ND: {df['ND'].max():,} MW"
    )

    print("\nActual/forecast indicator:")

    print(
        df["FORECAST_ACTUAL_INDICATOR"]
        .value_counts(dropna=False)
    )

def process_demand_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df["SETTLEMENT_DATE"] = parse_settlement_dates(
    df["SETTLEMENT_DATE"]
    )

    df = add_settlement_timestamp(df)

    df = df.drop(
        columns=["_id"],
        errors="ignore",
    )

    df.columns = [
        column.lower()
        for column in df.columns
    ]

    df = df.rename(
      columns={
        "nd": "national_demand_mw",
        "tsd": "transmission_system_demand_mw",
        "england_wales_demand": "england_wales_demand_mw",
        "embedded_wind_generation": "embedded_wind_generation_mw",
        "embedded_wind_capacity": "embedded_wind_capacity_mw",
        "embedded_solar_generation": "embedded_solar_generation_mw",
        "embedded_solar_capacity": "embedded_solar_capacity_mw",
      }
    )

    df = df.sort_values(
        "settlement_start"
    ).reset_index(drop=True)

    return df

def save_dataframe(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

def parse_settlement_dates(
    series: pd.Series,
) -> pd.Series:
    dates = series.astype("string")

    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    iso_mask = dates.str.match(
        r"^\d{4}-\d{2}-\d{2}$"
    )

    four_digit_year_mask = dates.str.match(
        r"^\d{2}-[A-Za-z]{3}-\d{4}$"
    )

    two_digit_year_mask = dates.str.match(
        r"^\d{2}-[A-Za-z]{3}-\d{2}$"
    )

    parsed.loc[iso_mask] = pd.to_datetime(
        dates.loc[iso_mask],
        format="%Y-%m-%d",
    )

    parsed.loc[four_digit_year_mask] = pd.to_datetime(
        dates.loc[four_digit_year_mask],
        format="%d-%b-%Y",
    )

    parsed.loc[two_digit_year_mask] = pd.to_datetime(
        dates.loc[two_digit_year_mask],
        format="%d-%b-%y",
    )

    if parsed.isna().any():
        unparsed = dates[parsed.isna()].unique()

        raise ValueError(
            f"Could not parse settlement dates: {unparsed}"
        )

    return parsed