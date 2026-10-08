from pathlib import Path

import pandas as pd

from gridcast_uk.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from gridcast_uk.models.baselines import (
    add_seasonal_naive_prediction,
)


DATA_PATH = Path(
    "data/processed/"
    "historical_demand_2021_2026.parquet"
)


df = pd.read_parquet(DATA_PATH)


df = add_seasonal_naive_prediction(
    df=df,
    lag_days=1,
    prediction_column="previous_day_prediction_mw",
)


df = add_seasonal_naive_prediction(
    df=df,
    lag_days=7,
    prediction_column="previous_week_prediction_mw",
)


validation_df = df[
    df["source_year"] == 2025
].copy()


print(
    f"Validation rows: {len(validation_df):,}"
)

def evaluate(
    name: str,
    prediction_column: str,
) -> None:
    valid = validation_df[
        prediction_column
    ].notna()

    actual = validation_df.loc[
        valid,
        "national_demand_mw",
    ]

    prediction = validation_df.loc[
        valid,
        prediction_column,
    ]

    mae = mean_absolute_error(
        actual,
        prediction,
    )

    rmse = root_mean_squared_error(
        actual,
        prediction,
    )

    coverage = (
        valid.mean() * 100
    )

    print()
    print(name)
    print("-" * len(name))

    print(
        f"Forecasts available: "
        f"{valid.sum():,} / "
        f"{len(validation_df):,}"
    )

    print(
        f"Coverage: {coverage:.2f}%"
    )

    print(
        f"MAE: {mae:,.0f} MW"
    )

    print(
        f"RMSE: {rmse:,.0f} MW"
    )

evaluate(
    name="Previous-day baseline",
    prediction_column="previous_day_prediction_mw",
)


evaluate(
    name="Previous-week baseline",
    prediction_column="previous_week_prediction_mw",
)

validation_df["day_of_week"] = (
    validation_df["settlement_start"]
    .dt.day_name()
)

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def mae_by_weekday(
    prediction_column: str,
) -> pd.Series:
    valid = validation_df[
        prediction_column
    ].notna()

    evaluation_df = (
        validation_df.loc[
            valid,
            [
                "day_of_week",
                "national_demand_mw",
                prediction_column,
            ],
        ]
        .copy()
    )

    evaluation_df["absolute_error"] = (
        evaluation_df["national_demand_mw"]
        - evaluation_df[prediction_column]
    ).abs()

    return (
        evaluation_df
        .groupby("day_of_week")["absolute_error"]
        .mean()
        .reindex(weekday_order)
    )


previous_day_by_weekday = mae_by_weekday(
    "previous_day_prediction_mw"
)

previous_week_by_weekday = mae_by_weekday(
    "previous_week_prediction_mw"
)


print("\nPrevious-day MAE by weekday:")
print(previous_day_by_weekday.round(0))

print("\nPrevious-week MAE by weekday:")
print(previous_week_by_weekday.round(0))

def mae_by_weekday(
    data: pd.DataFrame,
    prediction_column: str,
) -> pd.Series:
    valid = data[prediction_column].notna()

    evaluation_df = data.loc[
        valid,
        [
            "day_of_week",
            "national_demand_mw",
            prediction_column,
        ],
    ].copy()

    evaluation_df["absolute_error"] = (
        evaluation_df["national_demand_mw"]
        - evaluation_df[prediction_column]
    ).abs()

    return (
        evaluation_df
        .groupby("day_of_week")["absolute_error"]
        .mean()
        .reindex(weekday_order)
    )

train_df = df[
    df["source_year"].between(2021, 2024)
].copy()

train_df["day_of_week"] = (
    train_df["settlement_start"]
    .dt.day_name()
)

train_previous_day = mae_by_weekday(
    train_df,
    "previous_day_prediction_mw",
)

train_previous_week = mae_by_weekday(
    train_df,
    "previous_week_prediction_mw",
)

comparison = pd.DataFrame(
    {
        "previous_day": train_previous_day,
        "previous_week": train_previous_week,
    }
)

comparison["best_baseline"] = comparison.idxmin(
    axis=1
)

print(comparison)

best_by_weekday = comparison[
    "best_baseline"
].to_dict()

def choose_hybrid_prediction(row):
    baseline = best_by_weekday[
        row["day_of_week"]
    ]

    if baseline == "previous_day":
        return row[
            "previous_day_prediction_mw"
        ]

    return row[
        "previous_week_prediction_mw"
    ]

validation_df[
    "hybrid_prediction_mw"
] = validation_df.apply(
    choose_hybrid_prediction,
    axis=1,
)

evaluate(
    name="Weekday-adaptive baseline",
    prediction_column="hybrid_prediction_mw",
)

candidate_weights = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

weight_results = []

valid_train = (
    train_df["previous_day_prediction_mw"].notna()
    & train_df["previous_week_prediction_mw"].notna()
)

blend_train = train_df.loc[valid_train].copy()

for weight in candidate_weights:
    prediction = (
        weight
        * blend_train["previous_day_prediction_mw"]
        + (1 - weight)
        * blend_train["previous_week_prediction_mw"]
    )

    mae = mean_absolute_error(
        blend_train["national_demand_mw"],
        prediction,
    )

    weight_results.append(
        {
            "weight_previous_day": weight,
            "weight_previous_week": 1 - weight,
            "mae_mw": mae,
        }
    )

weight_results_df = pd.DataFrame(
    weight_results
)

print("\nTraining blend results:")
print(weight_results_df.round(1))

best_row = weight_results_df.loc[
    weight_results_df["mae_mw"].idxmin()
]

best_weight = best_row[
    "weight_previous_day"
]

print(
    f"\nBest previous-day weight: "
    f"{best_weight:.1f}"
)

validation_df["weighted_prediction_mw"] = (
    best_weight
    * validation_df["previous_day_prediction_mw"]
    + (1 - best_weight)
    * validation_df["previous_week_prediction_mw"]
)

evaluate(
    name="Weighted baseline",
    prediction_column="weighted_prediction_mw",
)