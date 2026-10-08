from pathlib import Path

import pandas as pd

from gridcast_uk.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from gridcast_uk.models.gradient_boosting import (
    build_gradient_boosting_model,
)


DATA_PATH = Path(
    "data/processed/"
    "demand_features_with_regional_weather_2024_2025.parquet"
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
# National-average weather features
# --------------------------------------------------

NATIONAL_WEATHER_FEATURES = [
    "forecast_temperature_c",
    "forecast_shortwave_radiation",
    "forecast_cloud_cover_pct",
    "forecast_wind_speed",
]


# --------------------------------------------------
# Regional weather features
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


# --------------------------------------------------
# Complete feature sets
# --------------------------------------------------

NATIONAL_FEATURES = (
    CONTROL_FEATURES
    + NATIONAL_WEATHER_FEATURES
)

REGIONAL_FEATURES = (
    CONTROL_FEATURES
    + REGIONAL_WEATHER_FEATURES
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_parquet(
    DATA_PATH
)


# --------------------------------------------------
# Use exactly the same rows for every model
# --------------------------------------------------

required_columns = list(
    dict.fromkeys(
        CONTROL_FEATURES
        + NATIONAL_WEATHER_FEATURES
        + REGIONAL_WEATHER_FEATURES
        + [TARGET]
    )
)

df = df.dropna(
    subset=required_columns
)


# --------------------------------------------------
# Same matched time split
#
# Train:
# 7 March 2024 -> 31 December 2024
#
# Validate:
# 2025
# --------------------------------------------------

train_df = df[
    df["source_year"] == 2024
].copy()

validation_df = df[
    df["source_year"] == 2025
].copy()


print(
    f"Training rows: {len(train_df):,}"
)

print(
    f"Validation rows: {len(validation_df):,}"
)

print(
    f"National model features: "
    f"{len(NATIONAL_FEATURES)}"
)

print(
    f"Regional model features: "
    f"{len(REGIONAL_FEATURES)}"
)


# --------------------------------------------------
# Frozen gradient boosting configuration
# --------------------------------------------------

def build_model():
    return build_gradient_boosting_model(
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_iter=400,
    )


# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------

def evaluate_model(
    name: str,
    features: list[str],
) -> tuple[float, float]:

    model = build_model()

    model.fit(
        train_df[features],
        train_df[TARGET],
    )

    predictions = model.predict(
        validation_df[features]
    )

    actual = (
        validation_df[TARGET]
        .reset_index(drop=True)
    )

    predictions = pd.Series(
        predictions
    )

    mae = mean_absolute_error(
        actual,
        predictions,
    )

    rmse = root_mean_squared_error(
        actual,
        predictions,
    )

    print(
        f"\n--- {name} ---"
    )

    print(
        f"MAE: {mae:,.0f} MW"
    )

    print(
        f"RMSE: {rmse:,.0f} MW"
    )

    return mae, rmse


# --------------------------------------------------
# National-average weather model
# --------------------------------------------------

national_mae, national_rmse = evaluate_model(
    name="National-average weather model",
    features=NATIONAL_FEATURES,
)


# --------------------------------------------------
# Regional weather model
# --------------------------------------------------

regional_mae, regional_rmse = evaluate_model(
    name="Regional weather model",
    features=REGIONAL_FEATURES,
)


# --------------------------------------------------
# Improvement from preserving geography
# --------------------------------------------------

mae_improvement_mw = (
    national_mae
    - regional_mae
)

rmse_improvement_mw = (
    national_rmse
    - regional_rmse
)


mae_improvement_pct = (
    mae_improvement_mw
    / national_mae
    * 100
)

rmse_improvement_pct = (
    rmse_improvement_mw
    / national_rmse
    * 100
)


print(
    "\n--- Regional weather improvement ---"
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