from pathlib import Path

import numpy as np
import pandas as pd

from gridcast_uk.evaluation.metrics import (
    mean_absolute_error,
)
from gridcast_uk.models.gradient_boosting import (
    build_gradient_boosting_model,
)


DATA_PATH = Path(
    "data/processed/"
    "demand_features_with_regional_weather_2024_2025.parquet"
)

TARGET = "national_demand_mw"


CONTROL_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_england_wales_holiday",
    "is_scotland_holiday",
    "demand_lag_1d_mw",
    "demand_lag_2d_mw",
    "demand_lag_7d_mw",
    "demand_change_1d_mw",
]


LOCATIONS = [
    "birmingham",
    "cardiff",
    "edinburgh",
    "glasgow",
    "leeds",
    "london",
    "manchester",
    "newcastle",
]


WEATHER_TYPES = [
    "temperature_c",
    "shortwave_radiation",
    "cloud_cover_pct",
    "wind_speed",
]


REGIONAL_WEATHER_FEATURES = [
    f"{location}_{weather_type}"
    for location in LOCATIONS
    for weather_type in WEATHER_TYPES
]


FEATURES = (
    CONTROL_FEATURES
    + REGIONAL_WEATHER_FEATURES
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_parquet(
    DATA_PATH
)

df = df.dropna(
    subset=FEATURES + [TARGET]
)


train_df = df[
    df["source_year"] == 2024
].copy()

validation_df = df[
    df["source_year"] == 2025
].copy()


# --------------------------------------------------
# Train frozen regional model
# --------------------------------------------------

model = build_gradient_boosting_model(
    learning_rate=0.05,
    max_leaf_nodes=15,
    max_iter=400,
)

model.fit(
    train_df[FEATURES],
    train_df[TARGET],
)


X_validation = (
    validation_df[FEATURES]
    .reset_index(drop=True)
)

y_validation = (
    validation_df[TARGET]
    .reset_index(drop=True)
)


baseline_predictions = model.predict(
    X_validation
)

baseline_mae = mean_absolute_error(
    y_validation,
    pd.Series(baseline_predictions),
)


print(
    f"Baseline regional-model MAE: "
    f"{baseline_mae:,.1f} MW"
)


# --------------------------------------------------
# Grouped permutation importance
#
# All columns in a group are shuffled together.
# This destroys that group's relationship with demand
# while preserving relationships within the group.
# --------------------------------------------------

def grouped_permutation_importance(
    group_columns: list[str],
    repeats: int = 5,
) -> float:

    mae_increases = []

    rng = np.random.default_rng(
        42
    )

    for _ in range(repeats):

        shuffled = (
            X_validation.copy()
        )

        permutation = rng.permutation(
            len(shuffled)
        )

        shuffled[
            group_columns
        ] = (
            shuffled[
                group_columns
            ]
            .iloc[permutation]
            .to_numpy()
        )

        predictions = model.predict(
            shuffled
        )

        shuffled_mae = (
            mean_absolute_error(
                y_validation,
                pd.Series(predictions),
            )
        )

        mae_increases.append(
            shuffled_mae
            - baseline_mae
        )

    return float(
        np.mean(mae_increases)
    )


# --------------------------------------------------
# Importance by geographic location
# --------------------------------------------------

location_results = []


for location in LOCATIONS:

    columns = [
        f"{location}_{weather_type}"
        for weather_type in WEATHER_TYPES
    ]

    importance = (
        grouped_permutation_importance(
            columns
        )
    )

    location_results.append(
        {
            "location": location,
            "mae_increase_mw": importance,
        }
    )


location_df = (
    pd.DataFrame(
        location_results
    )
    .sort_values(
        "mae_increase_mw",
        ascending=False,
    )
)


print(
    "\n--- Regional weather importance by location ---"
)

print(
    location_df.to_string(
        index=False,
        float_format=lambda x: f"{x:,.1f}",
    )
)


# --------------------------------------------------
# Importance by weather variable
# --------------------------------------------------

weather_type_results = []


for weather_type in WEATHER_TYPES:

    columns = [
        f"{location}_{weather_type}"
        for location in LOCATIONS
    ]

    importance = (
        grouped_permutation_importance(
            columns
        )
    )

    weather_type_results.append(
        {
            "weather_type": weather_type,
            "mae_increase_mw": importance,
        }
    )


weather_type_df = (
    pd.DataFrame(
        weather_type_results
    )
    .sort_values(
        "mae_increase_mw",
        ascending=False,
    )
)


print(
    "\n--- Regional weather importance by weather type ---"
)

print(
    weather_type_df.to_string(
        index=False,
        float_format=lambda x: f"{x:,.1f}",
    )
)