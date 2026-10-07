from sklearn.ensemble import HistGradientBoostingRegressor


def build_gradient_boosting_model(
    learning_rate: float = 0.1,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        early_stopping=False,
        random_state=42,
    )