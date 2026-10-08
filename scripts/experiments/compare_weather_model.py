from pathlib import Path

import pandas as pd

from gridcast_uk.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from gridcast_uk.models.gradient_boosting import (
    build_gradient_boosting_model,
)
from sklearn.inspection import permutation_importance

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


df = pd.read_parquet(
    DATA_PATH
)


# Use only rows where every weather feature is available.
# This guarantees both models are evaluated
# on exactly the same observations.
df = df.dropna(
    subset=WEATHER_FEATURES + [TARGET]
)


# Train on the weather-covered portion of 2024.
train_df = df[
    df["source_year"] == 2024
].copy()

# Validate on 2025.
validation_df = df[
    df["source_year"] == 2025
].copy()


print(
    f"Training rows: {len(train_df):,}"
)

print(
    f"Validation rows: {len(validation_df):,}"
)


def build_model():
    return build_gradient_boosting_model(
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_iter=400,
    )


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

    if name == "Weather model":
        importance = permutation_importance(
            model,
            validation_df[features],
            validation_df[TARGET],
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=42,
        )

        importance_df = pd.DataFrame(
            {
                "feature": features,
                "mae_increase_mw": (
                    importance.importances_mean
                ),
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "mae_increase_mw",
                ascending=False,
            )
        )

        print(
            "\n--- Weather model permutation importance ---"
        )

        print(
            importance_df.to_string(
                index=False,
                float_format=lambda x: f"{x:,.1f}",
            )
        )

    return mae, rmse


control_mae, control_rmse = evaluate_model(
    name="Control model",
    features=CONTROL_FEATURES,
)


weather_mae, weather_rmse = evaluate_model(
    name="Weather model",
    features=WEATHER_FEATURES,
)


mae_improvement = (
    (control_mae - weather_mae)
    / control_mae
    * 100
)

rmse_improvement = (
    (control_rmse - weather_rmse)
    / control_rmse
    * 100
)


print(
    "\n--- Weather improvement ---"
)

print(
    f"MAE improvement: "
    f"{mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)