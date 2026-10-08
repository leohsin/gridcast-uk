from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gridcast_uk.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)


PREDICTIONS_PATH = Path(
    "data/processed/"
    "final_test_predictions_2026.parquet"
)

FIGURE_DIR = Path(
    "reports/figures"
)

TABLE_DIR = Path(
    "reports/tables"
)

TARGET = "national_demand_mw"


# --------------------------------------------------
# Create output folders
# --------------------------------------------------

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Load saved final-test predictions
# --------------------------------------------------

df = pd.read_parquet(
    PREDICTIONS_PATH
)

df["settlement_start"] = pd.to_datetime(
    df["settlement_start"]
)


# --------------------------------------------------
# Add useful error columns
# --------------------------------------------------

df["control_error_mw"] = (
    df[TARGET]
    - df["control_prediction_mw"]
)

df["final_error_mw"] = (
    df[TARGET]
    - df["final_prediction_mw"]
)

df["month"] = (
    df["settlement_start"].dt.month
)

df["hour"] = (
    df["settlement_start"].dt.hour
    + df["settlement_start"].dt.minute / 60
)

df["date"] = (
    df["settlement_start"].dt.date
)


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

control_mae = mean_absolute_error(
    df[TARGET],
    df["control_prediction_mw"],
)

control_rmse = root_mean_squared_error(
    df[TARGET],
    df["control_prediction_mw"],
)

final_mae = mean_absolute_error(
    df[TARGET],
    df["final_prediction_mw"],
)

final_rmse = root_mean_squared_error(
    df[TARGET],
    df["final_prediction_mw"],
)

control_bias = (
    df["control_error_mw"].mean()
)

final_bias = (
    df["final_error_mw"].mean()
)


overall_metrics = pd.DataFrame(
    {
        "model": [
            "Control",
            "Regional weather",
        ],
        "mae_mw": [
            control_mae,
            final_mae,
        ],
        "rmse_mw": [
            control_rmse,
            final_rmse,
        ],
        "mean_error_mw": [
            control_bias,
            final_bias,
        ],
    }
)


print(
    "\n--- 2026 final test metrics ---"
)

print(
    overall_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:,.1f}",
    )
)


overall_metrics.to_csv(
    TABLE_DIR / "final_test_metrics_2026.csv",
    index=False,
)


# --------------------------------------------------
# Monthly MAE
# --------------------------------------------------

monthly_metrics = (
    df.groupby("month")
    .agg(
        control_mae_mw=(
            "control_absolute_error_mw",
            "mean",
        ),
        final_mae_mw=(
            "final_absolute_error_mw",
            "mean",
        ),
    )
)

monthly_metrics["improvement_mw"] = (
    monthly_metrics["control_mae_mw"]
    - monthly_metrics["final_mae_mw"]
)

monthly_metrics["improvement_pct"] = (
    monthly_metrics["improvement_mw"]
    / monthly_metrics["control_mae_mw"]
    * 100
)


print(
    "\n--- MAE by month ---"
)

print(
    monthly_metrics.round(1).to_string()
)


monthly_metrics.to_csv(
    TABLE_DIR / "final_test_monthly_metrics_2026.csv"
)


# --------------------------------------------------
# Half-hour MAE profile
# --------------------------------------------------

hourly_metrics = (
    df.groupby("hour")
    .agg(
        control_mae_mw=(
            "control_absolute_error_mw",
            "mean",
        ),
        final_mae_mw=(
            "final_absolute_error_mw",
            "mean",
        ),
    )
)

hourly_metrics["improvement_mw"] = (
    hourly_metrics["control_mae_mw"]
    - hourly_metrics["final_mae_mw"]
)

hourly_metrics["improvement_pct"] = (
    hourly_metrics["improvement_mw"]
    / hourly_metrics["control_mae_mw"]
    * 100
)


print(
    "\n--- MAE by half-hour ---"
)

print(
    hourly_metrics.round(1).to_string()
)


hourly_metrics.to_csv(
    TABLE_DIR / "final_test_hourly_metrics_2026.csv"
)


# --------------------------------------------------
# Daily MAE
# --------------------------------------------------

daily_metrics = (
    df.groupby("date")
    .agg(
        control_mae_mw=(
            "control_absolute_error_mw",
            "mean",
        ),
        final_mae_mw=(
            "final_absolute_error_mw",
            "mean",
        ),
    )
    .reset_index()
)

daily_metrics["date"] = pd.to_datetime(
    daily_metrics["date"]
)

daily_metrics["improvement_mw"] = (
    daily_metrics["control_mae_mw"]
    - daily_metrics["final_mae_mw"]
)


daily_metrics.to_csv(
    TABLE_DIR / "final_test_daily_metrics_2026.csv",
    index=False,
)


# --------------------------------------------------
# Figure 1:
# Actual vs predictions for first seven days
#
# We use the first seven days rather than selecting
# a particularly good or bad week after seeing the
# test results.
# --------------------------------------------------

start_time = (
    df["settlement_start"].min()
)

end_time = (
    start_time
    + pd.Timedelta(days=7)
)

first_week = df[
    (df["settlement_start"] >= start_time)
    & (df["settlement_start"] < end_time)
].copy()


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    first_week["settlement_start"],
    first_week[TARGET],
    label="Actual demand",
)

plt.plot(
    first_week["settlement_start"],
    first_week["control_prediction_mw"],
    label="Control",
)

plt.plot(
    first_week["settlement_start"],
    first_week["final_prediction_mw"],
    label="Regional weather",
)

plt.xlabel(
    "Settlement time"
)

plt.ylabel(
    "National demand (MW)"
)

plt.title(
    "GridCast UK: 2026 final test — first seven days"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "final_test_first_week_2026.png",
    dpi=200,
)

plt.close()


# --------------------------------------------------
# Figure 2:
# Monthly MAE comparison
# --------------------------------------------------

monthly_plot = (
    monthly_metrics[
        [
            "control_mae_mw",
            "final_mae_mw",
        ]
    ]
    .rename(
        columns={
            "control_mae_mw": "Control",
            "final_mae_mw": "Regional weather",
        }
    )
)


ax = monthly_plot.plot(
    kind="bar",
    figsize=(11, 6),
)

ax.set_xlabel(
    "Month"
)

ax.set_ylabel(
    "MAE (MW)"
)

ax.set_title(
    "2026 final test MAE by month"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "final_test_mae_by_month_2026.png",
    dpi=200,
)

plt.close()


# --------------------------------------------------
# Figure 3:
# Error profile throughout the day
# --------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    hourly_metrics.index,
    hourly_metrics["control_mae_mw"],
    label="Control",
)

plt.plot(
    hourly_metrics.index,
    hourly_metrics["final_mae_mw"],
    label="Regional weather",
)

plt.xlabel(
    "Hour of day"
)

plt.ylabel(
    "MAE (MW)"
)

plt.title(
    "2026 final test MAE by settlement time"
)

plt.xticks(
    range(0, 24, 2)
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "final_test_mae_by_hour_2026.png",
    dpi=200,
)

plt.close()


# --------------------------------------------------
# Figure 4:
# Daily final-model MAE through the test period
# --------------------------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    daily_metrics["date"],
    daily_metrics["final_mae_mw"],
)

plt.axhline(
    final_mae,
    linestyle="--",
    label=(
        f"Overall MAE "
        f"({final_mae:,.0f} MW)"
    ),
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Daily MAE (MW)"
)

plt.title(
    "Regional-weather model daily error — 2026 final test"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "final_test_daily_mae_2026.png",
    dpi=200,
)

plt.close()


print(
    "\nSaved reporting tables to:"
)

print(
    TABLE_DIR
)

print(
    "\nSaved figures to:"
)

print(
    FIGURE_DIR
)