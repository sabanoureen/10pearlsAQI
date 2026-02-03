from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.base import RegressorMixin

for model in [rf, xgb, gb]:
    assert isinstance(model, RegressorMixin)



def train_ensemble(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:squarederror",
        random_state=42
    )

    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    ensemble = VotingRegressor(
        estimators=[
            ("rf", rf),
            ("xgb", xgb),
            ("gb", gb),
        ]
    )

    ensemble.fit(X_train, y_train)
    return ensemble
