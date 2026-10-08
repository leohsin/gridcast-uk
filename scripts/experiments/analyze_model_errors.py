from pathlib import Path

import pandas as pd

from gridcast_uk.models.gradient_boosting import (
    build_gradient_boosting_model,
)


DATA_PATH = Path(
    "data/processed/"
    "demand_features_2021_2026.parquet"
)

FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "demand_lag_1d_mw",
    "demand_lag_2d_mw",
    "demand_lag_7d_mw",
    "demand_change_1d_mw",
]

TARGET = "national_demand_mw"


# Load feature dataset
df = pd.read_parquet(DATA_PATH)


# Same train / validation split used previously
train_df = df[
    df["source_year"].between(2021, 2024)
].copy()

validation_df = df[
    df["source_year"] == 2025
].copy()


# Remove rows where required features are unavailable
train_df = train_df.dropna(
    subset=FEATURES + [TARGET]
)

validation_df = validation_df.dropna(
    subset=FEATURES + [TARGET]
)


# Prepare X and y
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]


# Frozen tuned model
model = build_gradient_boosting_model(
    learning_rate=0.05,
    max_leaf_nodes=15,
    max_iter=400,
)


# Train on 2021-2024
model.fit(
    X_train,
    y_train,
)


# Predict 2025
predictions = model.predict(
    X_validation
)


# Build error-analysis table
evaluation_df = validation_df[
    [
        "settlement_start",
        "hour",
        "day_of_week",
        "month",
        "national_demand_mw",
        "demand_lag_1d_mw",
        "demand_lag_2d_mw",
        "demand_lag_7d_mw",
    ]
].copy()

evaluation_df["prediction_mw"] = predictions

evaluation_df["error_mw"] = (
    evaluation_df["national_demand_mw"]
    - evaluation_df["prediction_mw"]
)

evaluation_df["absolute_error_mw"] = (
    evaluation_df["error_mw"].abs()
)


# -------------------------
# MAE by weekday
# -------------------------

weekday_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

evaluation_df["weekday"] = (
    evaluation_df["day_of_week"]
    .map(weekday_names)
)

weekday_errors = (
    evaluation_df
    .groupby("weekday")["absolute_error_mw"]
    .mean()
    .reindex(weekday_order)
)

print("\n--- MAE by weekday ---")
print(weekday_errors.round(0))


# -------------------------
# MAE by month
# -------------------------

monthly_errors = (
    evaluation_df
    .groupby("month")["absolute_error_mw"]
    .mean()
)

print("\n--- MAE by month ---")
print(monthly_errors.round(0))


# -------------------------
# MAE by hour
# -------------------------

hourly_errors = (
    evaluation_df
    .groupby("hour")["absolute_error_mw"]
    .mean()
)

print("\n--- MAE by hour ---")
print(hourly_errors.round(0))


# -------------------------
# Largest individual errors
# -------------------------

worst_errors = (
    evaluation_df
    .sort_values(
        "absolute_error_mw",
        ascending=False,
    )
    .head(20)
)

print("\n--- 20 largest errors ---")

print(
    worst_errors[
        [
            "settlement_start",
            "national_demand_mw",
            "prediction_mw",
            "error_mw",
            "absolute_error_mw",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.0f}",
    )
)

evaluation_df["date"] = (
    evaluation_df["settlement_start"].dt.date
)

daily_errors = (
    evaluation_df
    .groupby("date")
    .agg(
        mae_mw=(
            "absolute_error_mw",
            "mean",
        ),
        mean_error_mw=(
            "error_mw",
            "mean",
        ),
        max_error_mw=(
            "absolute_error_mw",
            "max",
        ),
        actual_mean_mw=(
            "national_demand_mw",
            "mean",
        ),
        prediction_mean_mw=(
            "prediction_mw",
            "mean",
        ),
    )
    .sort_values(
        "mae_mw",
        ascending=False,
    )
)

print("\n--- 20 worst forecast days ---")

print(
    daily_errors
    .head(20)
    .round(0)
    .to_string()
)

march_18 = evaluation_df[
    evaluation_df["settlement_start"].dt.date
    == pd.Timestamp("2025-03-18").date()
]

print("\n--- March 18, 2025 ---")

print(
    march_18[
        [
            "settlement_start",
            "national_demand_mw",
            "prediction_mw",
            "demand_lag_1d_mw",
            "demand_lag_2d_mw",
            "demand_lag_7d_mw",
            "error_mw",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.0f}",
    )
)

dates_to_check = pd.to_datetime(
    [
        "2025-03-11",
        "2025-03-17",
        "2025-03-18",
    ]
).date

solar_check = df[
    df["settlement_start"]
    .dt.date
    .isin(dates_to_check)
][
    [
        "settlement_start",
        "national_demand_mw",
        "embedded_solar_generation_mw",
        "embedded_wind_generation_mw",
    ]
].copy()

solar_check = solar_check[
    solar_check["settlement_start"].dt.hour.between(
        8,
        18,
    )
]

print(
    solar_check.to_string(index=False)
)

solar_check["date"] = (
    solar_check["settlement_start"].dt.date
)

print(
    solar_check
    .groupby("date")
    .agg(
        mean_demand_mw=(
            "national_demand_mw",
            "mean",
        ),
        max_solar_mw=(
            "embedded_solar_generation_mw",
            "max",
        ),
        mean_solar_mw=(
            "embedded_solar_generation_mw",
            "mean",
        ),
        mean_wind_mw=(
            "embedded_wind_generation_mw",
            "mean",
        ),
    )
    .round(0)
)