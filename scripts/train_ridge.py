from pathlib import Path

import pandas as pd

from gridcast_uk.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from gridcast_uk.models.ridge import (
    build_ridge_model,
)


DATA_PATH = Path(
    "data/processed/"
    "demand_features_2021_2026.parquet"
)


df = pd.read_parquet(DATA_PATH)

FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "demand_lag_1d_mw",
    "demand_lag_2d_mw",
    "demand_lag_7d_mw",
    "demand_change_1d_mw",
]

TARGET = "national_demand_mw"

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

model = build_ridge_model(
    alpha=100.0
)

model.fit(
    X_train,
    y_train,
)

validation_predictions = model.predict(
    X_validation
)

mae = mean_absolute_error(
    y_validation.reset_index(drop=True),
    pd.Series(validation_predictions),
)

rmse = root_mean_squared_error(
    y_validation.reset_index(drop=True),
    pd.Series(validation_predictions),
)

BASELINE_WEIGHT_PREVIOUS_DAY = 0.6

validation_df["weighted_baseline_mw"] = (
    BASELINE_WEIGHT_PREVIOUS_DAY
    * validation_df["demand_lag_1d_mw"]
    + (1 - BASELINE_WEIGHT_PREVIOUS_DAY)
    * validation_df["demand_lag_7d_mw"]
)

baseline_mae = mean_absolute_error(
    y_validation.reset_index(drop=True),
    validation_df[
        "weighted_baseline_mw"
    ].reset_index(drop=True),
)

baseline_rmse = root_mean_squared_error(
    y_validation.reset_index(drop=True),
    validation_df[
        "weighted_baseline_mw"
    ].reset_index(drop=True),
)

ridge_model = model.named_steps["ridge"]

coefficient_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "coefficient": ridge_model.coef_,
    }
)

coefficient_df["absolute_coefficient"] = (
    coefficient_df["coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    "absolute_coefficient",
    ascending=False,
)

print("\n--- Ridge coefficients ---")
print(
    coefficient_df[
        ["feature", "coefficient"]
    ].to_string(index=False)
)

print(
    f"\nIntercept: "
    f"{ridge_model.intercept_:,.0f} MW"
)

print("--- Ridge regression ---")

print(
    f"Training rows: {len(train_df):,}"
)

print(
    f"Validation rows: {len(validation_df):,}"
)

print(
    f"MAE: {mae:,.0f} MW"
)

print(
    f"RMSE: {rmse:,.0f} MW"
)

print("\n--- Weighted baseline on same rows ---")
print(f"MAE: {baseline_mae:,.0f} MW")
print(f"RMSE: {baseline_rmse:,.0f} MW")

mae_improvement = (
    (baseline_mae - mae)
    / baseline_mae
    * 100
)

rmse_improvement = (
    (baseline_rmse - rmse)
    / baseline_rmse
    * 100
)

print("\n--- Ridge improvement ---")
print(
    f"MAE improvement: "
    f"{mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)
