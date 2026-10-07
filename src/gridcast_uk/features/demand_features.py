import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    timestamp = df["settlement_start"]

    df["hour"] = (
        timestamp.dt.hour
        + timestamp.dt.minute / 60
    )

    df["day_of_week"] = timestamp.dt.dayofweek

    df["month"] = timestamp.dt.month

    df["is_weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
    2 * np.pi * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
    2 * np.pi * (df["month"] - 1) / 12
    )
    
    return df

def add_demand_lag(
    df: pd.DataFrame,
    lag_days: int,
    column_name: str,
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
            "national_demand_mw": column_name
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

def build_demand_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df = add_calendar_features(df)

    df = add_demand_lag(
        df=df,
        lag_days=1,
        column_name="demand_lag_1d_mw",
    )

    df = add_demand_lag(
        df=df,
        lag_days=2,
        column_name="demand_lag_2d_mw",
    )

    df = add_demand_lag(
        df=df,
        lag_days=7,
        column_name="demand_lag_7d_mw",
    )

    df["demand_change_1d_mw"] = (
        df["demand_lag_1d_mw"]
        - df["demand_lag_2d_mw"]
    )

    return df