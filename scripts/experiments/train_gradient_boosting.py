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
    "demand_features_2021_2026.parquet"
)

TARGET = "national_demand_mw"

FEATURES = [
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

df = pd.read_parquet(DATA_PATH)

train_df = df[
    df["source_year"].between(2021, 2024)
].copy()

validation_df = df[
    df["source_year"] == 2025
].copy()


train_df = train_df.dropna(
    subset=FEATURES + [TARGET]
)

validation_df = validation_df.dropna(
    subset=FEATURES + [TARGET]
)


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

model = build_gradient_boosting_model(
    learning_rate=0.05,
    max_leaf_nodes=15,
    max_iter=400,
)

model.fit(
    X_train,
    y_train,
)

predictions = model.predict(
    X_validation
)

mae = mean_absolute_error(
    y_validation.reset_index(drop=True),
    pd.Series(predictions),
)

rmse = root_mean_squared_error(
    y_validation.reset_index(drop=True),
    pd.Series(predictions),
)


print("--- Gradient boosting ---")
print(f"Training rows: {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")
print(f"MAE: {mae:,.0f} MW")
print(f"RMSE: {rmse:,.0f} MW")