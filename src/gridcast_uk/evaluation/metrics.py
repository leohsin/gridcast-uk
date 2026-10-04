import numpy as np
import pandas as pd


def mean_absolute_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    valid = y_true.notna() & y_pred.notna()

    if not valid.any():
        raise ValueError(
            "No valid observations available for evaluation."
        )

    errors = (
        y_true[valid] - y_pred[valid]
    ).abs()

    return float(errors.mean())


def root_mean_squared_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    valid = y_true.notna() & y_pred.notna()

    if not valid.any():
        raise ValueError(
            "No valid observations available for evaluation."
        )

    squared_errors = (
        y_true[valid] - y_pred[valid]
    ) ** 2

    return float(
        np.sqrt(squared_errors.mean())
    )