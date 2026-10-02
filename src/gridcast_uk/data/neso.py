from typing import Any
from pathlib import Path

import httpx
import pandas as pd

API_URL = (
    "https://" +
    "api.neso.energy" +
    "/api/3/action/datastore_search"
)

HISTORIC_DEMAND_2026_RESOURCE_ID = (
    "8a4a771c-3929-4e56-93ad-cdf13219dea5"
)


def fetch_demand_records(
    limit: int = 1000,
    offset: int = 0,
) -> dict[str, Any]:
    params = {
        "resource_id": HISTORIC_DEMAND_2026_RESOURCE_ID,
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
        raise RuntimeError("NESO API request was unsuccessful.")

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
    page_size: int = 1000,
) -> list[dict[str, Any]]:
    all_records = []

    offset = 0

    while True:
        data = fetch_demand_records(
            limit=page_size,
            offset=offset,
        )

        result = data["result"]

        records = result["records"]
        total = result["total"]

        all_records.extend(records)

        print(
            f"Downloaded {len(all_records)} "
            f"of {total} records..."
        )

        if len(all_records) >= total:
            break

        offset += page_size

    return all_records

def fetch_all_demand_data(
    page_size: int = 1000,
) -> pd.DataFrame:
    records = fetch_all_demand_records(
        page_size=page_size,
    )

    return pd.DataFrame(records)

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