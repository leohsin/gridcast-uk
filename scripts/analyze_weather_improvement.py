from pathlib import Path

import pandas as pd

from gridcast_uk.models.gradient_boosting import (
    build_gradient_boosting_model,
)


DATA_PATH = Path(
    "data/processed/"
    "demand_features_with_weather_2024_2025.parquet"
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


WEATHER_FEATURES = CONTROL_FEATURES + [
    "forecast_temperature_c",
    "forecast_shortwave_radiation",
    "forecast_cloud_cover_pct",
    "forecast_wind_speed",
]


df = pd.read_parquet(DATA_PATH)

df = df.dropna(
    subset=WEATHER_FEATURES + [TARGET]
)


train_df = df[
    df["source_year"] == 2024
].copy()

validation_df = df[
    df["source_year"] == 2025
].copy()


def build_model():
    return build_gradient_boosting_model(
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_iter=400,
    )


# -------------------------
# Train control model
# -------------------------

control_model = build_model()

control_model.fit(
    train_df[CONTROL_FEATURES],
    train_df[TARGET],
)

control_predictions = control_model.predict(
    validation_df[CONTROL_FEATURES]
)


# -------------------------
# Train weather model
# -------------------------

weather_model = build_model()

weather_model.fit(
    train_df[WEATHER_FEATURES],
    train_df[TARGET],
)

weather_predictions = weather_model.predict(
    validation_df[WEATHER_FEATURES]
)


# -------------------------
# Build comparison table
# -------------------------

evaluation_df = validation_df[
    [
        "settlement_start",
        "hour",
        "month",
        TARGET,
        "forecast_temperature_c",
        "forecast_shortwave_radiation",
        "forecast_cloud_cover_pct",
        "forecast_wind_speed",
    ]
].copy()


evaluation_df["control_prediction_mw"] = (
    control_predictions
)

evaluation_df["weather_prediction_mw"] = (
    weather_predictions
)


evaluation_df["control_absolute_error_mw"] = (
    evaluation_df[TARGET]
    - evaluation_df["control_prediction_mw"]
).abs()


evaluation_df["weather_absolute_error_mw"] = (
    evaluation_df[TARGET]
    - evaluation_df["weather_prediction_mw"]
).abs()


evaluation_df["error_reduction_mw"] = (
    evaluation_df["control_absolute_error_mw"]
    - evaluation_df["weather_absolute_error_mw"]
)


# -------------------------
# Improvement by hour
# -------------------------

hourly_comparison = (
    evaluation_df
    .groupby("hour")
    .agg(
        control_mae_mw=(
            "control_absolute_error_mw",
            "mean",
        ),
        weather_mae_mw=(
            "weather_absolute_error_mw",
            "mean",
        ),
    )
)

hourly_comparison["improvement_mw"] = (
    hourly_comparison["control_mae_mw"]
    - hourly_comparison["weather_mae_mw"]
)

hourly_comparison["improvement_pct"] = (
    hourly_comparison["improvement_mw"]
    / hourly_comparison["control_mae_mw"]
    * 100
)


print(
    "\n--- Control vs weather MAE by hour ---"
)

print(
    hourly_comparison
    .round(1)
    .to_string()
)


# -------------------------
# Improvement by month
# -------------------------

monthly_comparison = (
    evaluation_df
    .groupby("month")
    .agg(
        control_mae_mw=(
            "control_absolute_error_mw",
            "mean",
        ),
        weather_mae_mw=(
            "weather_absolute_error_mw",
            "mean",
        ),
    )
)

monthly_comparison["improvement_mw"] = (
    monthly_comparison["control_mae_mw"]
    - monthly_comparison["weather_mae_mw"]
)

monthly_comparison["improvement_pct"] = (
    monthly_comparison["improvement_mw"]
    / monthly_comparison["control_mae_mw"]
    * 100
)


print(
    "\n--- Control vs weather MAE by month ---"
)

print(
    monthly_comparison
    .round(1)
    .to_string()
)


# -------------------------
# March 18 investigation
# -------------------------

march_18 = evaluation_df[
    evaluation_df["settlement_start"]
    .dt.date
    == pd.Timestamp(
        "2025-03-18"
    ).date()
].copy()


print(
    "\n--- March 18, 2025 ---"
)

print(
    march_18[
        [
            "settlement_start",
            TARGET,
            "control_prediction_mw",
            "weather_prediction_mw",
            "control_absolute_error_mw",
            "weather_absolute_error_mw",
            "forecast_shortwave_radiation",
            "forecast_temperature_c",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.0f}",
    )
)


print(
    "\n--- March 18 summary ---"
)

print(
    f"Control MAE: "
    f"{march_18['control_absolute_error_mw'].mean():,.0f} MW"
)

print(
    f"Weather MAE: "
    f"{march_18['weather_absolute_error_mw'].mean():,.0f} MW"
)

march_18_improvement = (
    1
    - (
        march_18["weather_absolute_error_mw"].mean()
        / march_18["control_absolute_error_mw"].mean()
    )
) * 100

print(
    f"Improvement: "
    f"{march_18_improvement:.2f}%"
)