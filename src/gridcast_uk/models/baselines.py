import pandas as pd


def add_seasonal_naive_prediction(
    df: pd.DataFrame,
    lag_days: int,
    prediction_column: str,
) -> pd.DataFrame:
    source = df[
        [
            "settlement_date",
            "settlement_period",
            "national_demand_mw",
        ]
    ].copy()

    source["settlement_date"] = (
        source["settlement_date"]
        + pd.Timedelta(days=lag_days)
    )

    source = source.rename(
        columns={
            "national_demand_mw":
                prediction_column
        }
    )

    result = df.merge(
        source,
        on=[
            "settlement_date",
            "settlement_period",
        ],
        how="left",
        validate="one_to_one",
    )

    return result