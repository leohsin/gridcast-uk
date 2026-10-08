from itertools import product
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

learning_rates = [
    0.05,
    0.1,
]

max_leaf_nodes_values = [
    15,
    31,
    63,
]

max_iter_values = [
    100,
    200,
    400,
]

product(
    learning_rates,
    max_leaf_nodes_values,
    max_iter_values,
)

results = []


for (
    learning_rate,
    max_leaf_nodes,
    max_iter,
) in product(
    learning_rates,
    max_leaf_nodes_values,
    max_iter_values,
):

    print(
        "\nTesting:"
        f" learning_rate={learning_rate},"
        f" max_leaf_nodes={max_leaf_nodes},"
        f" max_iter={max_iter}"
    )

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

        model = build_gradient_boosting_model(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            max_iter=max_iter,
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
                "learning_rate": learning_rate,
                "max_leaf_nodes": max_leaf_nodes,
                "max_iter": max_iter,
                "train_end_year": train_end_year,
                "validation_year": validation_year,
                "mae_mw": mae,
                "rmse_mw": rmse,
            }
        )

        print(
            f"  Validation {validation_year}: "
            f"MAE={mae:,.0f} MW"
        )

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby(
        [
            "learning_rate",
            "max_leaf_nodes",
            "max_iter",
        ]
    )
    .agg(
        mean_mae_mw=("mae_mw", "mean"),
        mean_rmse_mw=("rmse_mw", "mean"),
    )
    .reset_index()
    .sort_values("mean_mae_mw")
)


print(
    "\n--- Gradient boosting tuning results ---"
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

best = summary.iloc[0]

print("\n--- Best configuration ---")

print(
    f"Learning rate: "
    f"{best['learning_rate']}"
)

print(
    f"Max leaf nodes: "
    f"{int(best['max_leaf_nodes'])}"
)

print(
    f"Max iterations: "
    f"{int(best['max_iter'])}"
)

print(
    f"Mean MAE: "
    f"{best['mean_mae_mw']:,.1f} MW"
)

print(
    f"Mean RMSE: "
    f"{best['mean_rmse_mw']:,.1f} MW"
)
