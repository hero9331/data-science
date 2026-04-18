from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_datasets(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(root / "data" / "model_dataset.csv")
    latest = pd.read_csv(root / "data" / "latest_features.csv")
    return train, latest


def build_preprocessor() -> tuple[ColumnTransformer, list[str]]:
    categorical_features = ["origin_state", "dest_state"]
    numeric_features = [
        "year",
        "tons",
        "value",
        "tons_lag1",
        "value_lag1",
        "tons_growth",
        "value_growth",
        "distance_miles",
        "cpi_pc1",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )
    return preprocessor, categorical_features + numeric_features


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    predicted = np.clip(predicted, 0, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    root = repo_root()
    train_df, latest_df = load_datasets(root)
    preprocessor, feature_cols = build_preprocessor()

    test_year = int(train_df["year"].max())
    train_split = train_df[train_df["year"] < test_year].copy()
    test_split = train_df[train_df["year"] == test_year].copy()

    if train_split.empty or test_split.empty:
        raise ValueError("Training or test split is empty. Check the model_dataset.csv contents.")

    x_train = train_split[feature_cols]
    y_train = train_split["target_next_year_tons"]
    x_test = test_split[feature_cols]
    y_test = test_split["target_next_year_tons"]

    models = build_models(preprocessor)
    metrics_rows: list[dict[str, float | str | int]] = []
    prediction_frame = test_split[
        ["origin_state", "origin_state_name", "dest_state", "dest_state_name", "year", "target_next_year_tons"]
    ].copy()
    prediction_frame = prediction_frame.rename(
        columns={
            "year": "feature_year",
            "target_next_year_tons": "actual_next_year_tons",
        }
    )
    prediction_frame["actual_year"] = prediction_frame["feature_year"] + 1

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predicted = np.clip(model.predict(x_test), 0, None)
        scores = evaluate_predictions(y_test, predicted)
        metrics_rows.append(
            {
                "model": model_name,
                "train_year_start": int(train_split["year"].min()),
                "train_year_end": int(train_split["year"].max()),
                "test_year": test_year,
                "target_year": test_year + 1,
                **scores,
            }
        )
        prediction_frame[f"{model_name}_predicted_tons"] = predicted

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse").reset_index(drop=True)
    best_model_name = str(metrics_df.iloc[0]["model"])

    final_model = build_models(preprocessor)[best_model_name]
    final_model.fit(train_df[feature_cols], train_df["target_next_year_tons"])
    latest_predictions = np.clip(final_model.predict(latest_df[feature_cols]), 0, None)

    forecast_df = latest_df[
        ["origin_state", "origin_state_name", "dest_state", "dest_state_name", "year"]
    ].copy()
    forecast_df = forecast_df.rename(columns={"year": "feature_year"})
    forecast_df["prediction_year"] = forecast_df["feature_year"] + 1
    forecast_df["selected_model"] = best_model_name
    forecast_df["predicted_tons"] = latest_predictions
    forecast_df = forecast_df.sort_values("predicted_tons", ascending=False).reset_index(drop=True)

    data_dir = root / "data"
    metrics_path = data_dir / "model_metrics.csv"
    test_predictions_path = data_dir / "test_predictions.csv"
    forecast_path = data_dir / "forecast_2025.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(test_predictions_path, index=False, encoding="utf-8-sig")
    forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")

    print(f"saved metrics: {metrics_path}")
    print(metrics_df.to_string(index=False))
    print(f"saved test predictions: {test_predictions_path} rows={len(prediction_frame):,}")
    print(f"saved forecast: {forecast_path} rows={len(forecast_df):,}")
    print(f"selected model for latest forecast: {best_model_name}")


if __name__ == "__main__":
    main()
