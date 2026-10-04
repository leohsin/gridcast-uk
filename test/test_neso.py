import pandas as pd
import pytest

from gridcast_uk.data.neso import (
    add_settlement_timestamp,
    process_demand_data,
    validate_raw_demand_data,
    parse_settlement_dates,
)


def test_process_demand_data():
    raw_df = pd.DataFrame(
        {
            "_id": [1, 2],
            "SETTLEMENT_DATE": [
                "2026-01-01",
                "2026-01-01",
            ],
            "SETTLEMENT_PERIOD": [1, 2],
            "ND": [25107, 25881],
            "FORECAST_ACTUAL_INDICATOR": [
                "A",
                "A",
            ],
        }
    )

    processed_df = process_demand_data(raw_df)

    assert "_id" not in processed_df.columns

    assert (
        "national_demand_mw"
        in processed_df.columns
    )

    assert (
        "settlement_start"
        in processed_df.columns
    )

    assert len(processed_df) == 2

    assert (
        processed_df["national_demand_mw"].tolist()
        == [25107, 25881]
    )


def test_settlement_intervals_are_30_minutes():
    df = pd.DataFrame(
        {
            "SETTLEMENT_DATE": [
                pd.Timestamp("2026-01-01")
            ] * 48,
            "SETTLEMENT_PERIOD": range(1, 49),
        }
    )

    result = add_settlement_timestamp(df)

    differences = (
        result["SETTLEMENT_START"]
        .dt.tz_convert("UTC")
        .diff()
        .dropna()
    )

    assert (
        differences
        == pd.Timedelta(minutes=30)
    ).all()


def test_spring_dst_day_has_46_periods():
    df = pd.DataFrame(
        {
            "SETTLEMENT_DATE": [
                pd.Timestamp("2026-03-29")
            ] * 46,
            "SETTLEMENT_PERIOD": range(1, 47),
        }
    )

    result = add_settlement_timestamp(df)

    assert len(result) == 46

    assert (
        result["SETTLEMENT_START"]
        .notna()
        .all()
    )


def test_autumn_dst_day_has_50_periods():
    df = pd.DataFrame(
        {
            "SETTLEMENT_DATE": [
                pd.Timestamp("2026-10-25")
            ] * 50,
            "SETTLEMENT_PERIOD": range(1, 51),
        }
    )

    result = add_settlement_timestamp(df)

    assert len(result) == 50

    assert (
        result["SETTLEMENT_START"]
        .notna()
        .all()
    )


def test_validation_rejects_duplicates():
    df = pd.DataFrame(
        {
            "SETTLEMENT_DATE": [
                "2026-01-01",
                "2026-01-01",
            ],
            "SETTLEMENT_PERIOD": [1, 1],
            "ND": [25000, 25000],
            "FORECAST_ACTUAL_INDICATOR": [
                "A",
                "A",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_raw_demand_data(df)

def test_parse_historical_settlement_date_formats():
    dates = pd.Series(
        [
            "01-JAN-2021",
            "01-JAN-2022",
            "01-Jan-23",
            "2024-01-01",
            "2025-01-01",
            "2026-01-01",
        ]
    )

    result = parse_settlement_dates(dates)

    expected = pd.Series(
        pd.to_datetime(
            [
                "2021-01-01",
                "2022-01-01",
                "2023-01-01",
                "2024-01-01",
                "2025-01-01",
                "2026-01-01",
            ]
        )
    )

    pd.testing.assert_series_equal(
    result,
    expected,
    check_dtype=False,
    )