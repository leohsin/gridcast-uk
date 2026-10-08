from pathlib import Path

import pandas as pd

from gridcast_uk.data.neso import save_dataframe
from gridcast_uk.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from gridcast_uk.models.gradient_boosting import (
    build_gradient_boosting_model,
)


DEVELOPMENT_PATH = Path(
    "data/processed/"
    "demand_features_with_regional_weather_2024_2025.parquet"
)

TEST_PATH = Path(
    "data/processed/"
    "demand_features_with_regional_weather_2026.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "final_test_predictions_2026.parquet"
)

TARGET = "national_demand_mw"


# --------------------------------------------------
# Core demand/calendar features
# --------------------------------------------------

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


# --------------------------------------------------
# Frozen regional weather features
# --------------------------------------------------

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


FINAL_FEATURES = (
    CONTROL_FEATURES
    + REGIONAL_WEATHER_FEATURES
)


# --------------------------------------------------
# Load development and final test datasets
# --------------------------------------------------

development_df = pd.read_parquet(
    DEVELOPMENT_PATH
)

test_df = pd.read_parquet(
    TEST_PATH
)


# Development data:
# weather-covered 2024 + all available 2025.
development_df = development_df[
    development_df["source_year"].isin(
        [2024, 2025]
    )
].copy()


# Final untouched test:
# 2026 only.
test_df = test_df[
    test_df["source_year"] == 2026
].copy()


# --------------------------------------------------
# Use only complete observations
# --------------------------------------------------

required_columns = (
    FINAL_FEATURES
    + [TARGET]
)


development_df = development_df.dropna(
    subset=required_columns
)

test_df = test_df.dropna(
    subset=required_columns
)


print("--- Final test setup ---")

print(
    f"Development rows: "
    f"{len(development_df):,}"
)

print(
    f"Test rows: "
    f"{len(test_df):,}"
)

print(
    f"Final model features: "
    f"{len(FINAL_FEATURES)}"
)

print(
    f"Development range: "
    f"{development_df['settlement_start'].min()} "
    f"to "
    f"{development_df['settlement_start'].max()}"
)

print(
    f"Test range: "
    f"{test_df['settlement_start'].min()} "
    f"to "
    f"{test_df['settlement_start'].max()}"
)


# --------------------------------------------------
# Frozen model configuration
#
# Do not tune this using 2026.
# --------------------------------------------------

def build_model():
    return build_gradient_boosting_model(
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_iter=400,
    )


# --------------------------------------------------
# Helper for model evaluation
# --------------------------------------------------

def train_and_predict(
    features: list[str],
) -> tuple[pd.Series, float, float]:

    model = build_model()

    model.fit(
        development_df[features],
        development_df[TARGET],
    )

    predictions = model.predict(
        test_df[features]
    )

    predictions = pd.Series(
        predictions
    )

    actual = (
        test_df[TARGET]
        .reset_index(drop=True)
    )

    mae = mean_absolute_error(
        actual,
        predictions,
    )

    rmse = root_mean_squared_error(
        actual,
        predictions,
    )

    return predictions, mae, rmse


# --------------------------------------------------
# Control model
#
# Same boosting algorithm, but without weather.
# --------------------------------------------------

control_predictions, control_mae, control_rmse = (
    train_and_predict(
        CONTROL_FEATURES
    )
)


print(
    "\n--- 2026 control model ---"
)

print(
    f"MAE: {control_mae:,.0f} MW"
)

print(
    f"RMSE: {control_rmse:,.0f} MW"
)


# --------------------------------------------------
# Final selected regional-weather model
# --------------------------------------------------

final_predictions, final_mae, final_rmse = (
    train_and_predict(
        FINAL_FEATURES
    )
)


print(
    "\n--- 2026 FINAL regional weather model ---"
)

print(
    f"MAE: {final_mae:,.0f} MW"
)

print(
    f"RMSE: {final_rmse:,.0f} MW"
)


# --------------------------------------------------
# Improvement over matched control
# --------------------------------------------------

mae_improvement_mw = (
    control_mae
    - final_mae
)

rmse_improvement_mw = (
    control_rmse
    - final_rmse
)


mae_improvement_pct = (
    mae_improvement_mw
    / control_mae
    * 100
)

rmse_improvement_pct = (
    rmse_improvement_mw
    / control_rmse
    * 100
)


print(
    "\n--- Final test improvement ---"
)

print(
    f"MAE improvement: "
    f"{mae_improvement_mw:,.0f} MW "
    f"({mae_improvement_pct:.2f}%)"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement_mw:,.0f} MW "
    f"({rmse_improvement_pct:.2f}%)"
)


# --------------------------------------------------
# Save final test predictions
#
# This lets us create plots and reporting later
# without retraining the model.
# --------------------------------------------------

prediction_df = (
    test_df[
        [
            "settlement_start",
            TARGET,
        ]
    ]
    .reset_index(drop=True)
    .copy()
)


prediction_df[
    "control_prediction_mw"
] = control_predictions


prediction_df[
    "final_prediction_mw"
] = final_predictions


prediction_df[
    "control_absolute_error_mw"
] = (
    prediction_df[TARGET]
    - prediction_df[
        "control_prediction_mw"
    ]
).abs()


prediction_df[
    "final_absolute_error_mw"
] = (
    prediction_df[TARGET]
    - prediction_df[
        "final_prediction_mw"
    ]
).abs()


save_dataframe(
    prediction_df,
    OUTPUT_PATH,
)


print(
    f"\nSaved final predictions to: "
    f"{OUTPUT_PATH}"
)