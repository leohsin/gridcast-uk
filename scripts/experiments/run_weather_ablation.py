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
    "demand_features_with_weather_2024_2025.parquet"
)

TARGET = "national_demand_mw"


# --------------------------------------------------
# Core features available to every model
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
# Weather feature groups
# --------------------------------------------------

TEMPERATURE_FEATURES = [
    "forecast_temperature_c",
]

RADIATION_FEATURES = [
    "forecast_shortwave_radiation",
]

ALL_WEATHER_FEATURES = [
    "forecast_temperature_c",
    "forecast_shortwave_radiation",
    "forecast_cloud_cover_pct",
    "forecast_wind_speed",
]


# --------------------------------------------------
# Model configurations for the ablation study
# --------------------------------------------------

MODEL_FEATURES = {
    "Control": CONTROL_FEATURES,

    "Temperature only": (
        CONTROL_FEATURES
        + TEMPERATURE_FEATURES
    ),

    "Radiation only": (
        CONTROL_FEATURES
        + RADIATION_FEATURES
    ),

    "Temperature + radiation": (
        CONTROL_FEATURES
        + TEMPERATURE_FEATURES
        + RADIATION_FEATURES
    ),

    "All weather": (
        CONTROL_FEATURES
        + ALL_WEATHER_FEATURES
    ),
}


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_parquet(
    DATA_PATH
)


# IMPORTANT:
#
# Use rows where ALL weather variables are available,
# even for models that use only temperature or radiation.
#
# That means every model is trained and evaluated
# on exactly the same observations.
df = df.dropna(
    subset=(
        CONTROL_FEATURES
        + ALL_WEATHER_FEATURES
        + [TARGET]
    )
)


# --------------------------------------------------
# Same matched experiment as before
#
# Train: weather-covered portion of 2024
# Validate: 2025
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


# --------------------------------------------------
# Frozen model configuration
# --------------------------------------------------

def build_model():
    return build_gradient_boosting_model(
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_iter=400,
    )


# --------------------------------------------------
# Train and evaluate every feature set
# --------------------------------------------------

results = []


for model_name, features in MODEL_FEATURES.items():

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

    results.append(
        {
            "model": model_name,
            "feature_count": len(features),
            "mae_mw": mae,
            "rmse_mw": rmse,
        }
    )


results_df = pd.DataFrame(
    results
)


# --------------------------------------------------
# Calculate improvement relative to control model
# --------------------------------------------------

control_row = results_df[
    results_df["model"] == "Control"
].iloc[0]

control_mae = control_row["mae_mw"]
control_rmse = control_row["rmse_mw"]


results_df["mae_improvement_mw"] = (
    control_mae
    - results_df["mae_mw"]
)

results_df["mae_improvement_pct"] = (
    results_df["mae_improvement_mw"]
    / control_mae
    * 100
)


results_df["rmse_improvement_mw"] = (
    control_rmse
    - results_df["rmse_mw"]
)

results_df["rmse_improvement_pct"] = (
    results_df["rmse_improvement_mw"]
    / control_rmse
    * 100
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print(
    "\n--- Weather ablation results ---"
)

print(
    results_df.to_string(
        index=False,
        formatters={
            "mae_mw": lambda x: f"{x:,.0f}",
            "rmse_mw": lambda x: f"{x:,.0f}",
            "mae_improvement_mw":
                lambda x: f"{x:,.0f}",
            "mae_improvement_pct":
                lambda x: f"{x:.2f}%",
            "rmse_improvement_mw":
                lambda x: f"{x:,.0f}",
            "rmse_improvement_pct":
                lambda x: f"{x:.2f}%",
        },
    )
)