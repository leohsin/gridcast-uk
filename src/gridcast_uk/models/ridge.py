from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_model(
    alpha: float = 1.0,
) -> Pipeline:
    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(alpha=alpha),
            ),
        ]
    )

    return model