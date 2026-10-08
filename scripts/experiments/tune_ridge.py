from pathlib import Path

import pandas as pd

from gridcast_uk.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from gridcast_uk.models.ridge import build_ridge_model


DATA_PATH = Path(
    "data/processed/"
    "demand_features_2021_2026.parquet"
)


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


df = pd.read_parquet(DATA_PATH)

folds = [
    {
        "train_end_year": 2021,
        "validation_year": 2022,
    },
    {
        "train_end_year": 2022,
        "validation_year": 2023,
    },
    {
        "train_end_year": 2023,
        "validation_year": 2024,
    },
]

candidate_alphas = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

results = []


for alpha in candidate_alphas:
    for fold in folds:
        train_end_year = fold["train_end_year"]
        validation_year = fold["validation_year"]

        train_df = df[
            df["source_year"].between(
                2021,
                train_end_year,
            )
        ].copy()

        validation_df = df[
            df["source_year"] == validation_year
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
            alpha=alpha
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

        results.append(
            {
                "alpha": alpha,
                "train_end_year": train_end_year,
                "validation_year": validation_year,
                "mae_mw": mae,
                "rmse_mw": rmse,
            }
        )

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby("alpha")
    .agg(
        mean_mae_mw=("mae_mw", "mean"),
        mean_rmse_mw=("rmse_mw", "mean"),
    )
    .reset_index()
    .sort_values("mean_mae_mw")
)

print("\n--- Ridge tuning results ---")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

best_alpha = summary.iloc[0]["alpha"]

print(
    f"\nBest alpha: {best_alpha}"
)