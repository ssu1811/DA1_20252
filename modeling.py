from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from features import FEATURE_COLUMNS, TARGET_COLUMN


MODEL_OPTIONS = ["Linear Regression", "Random Forest", "SVR"]


@dataclass
class TrainingResult:
    model_name: str
    model: object
    metrics: dict[str, float | int | str]
    predictions: pd.DataFrame
    next_prediction: dict[str, float | str]


def make_model(model_name: str):
    if model_name == "Linear Regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
    if model_name == "SVR":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.01)),
            ]
        )
    raise ValueError(f"Mô hình chưa được hỗ trợ: {model_name}")


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def next_business_day(value: pd.Timestamp) -> pd.Timestamp:
    next_day = pd.Timestamp(value).normalize() + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def train_and_predict(feature_df: pd.DataFrame, model_name: str, test_size: float) -> TrainingResult:
    training_df = feature_df.dropna(subset=[TARGET_COLUMN]).copy()
    if len(training_df) < 80:
        raise ValueError("Cần ít nhất 80 dòng dữ liệu hợp lệ sau khi tạo đặc trưng.")

    split_index = int(len(training_df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(training_df):
        raise ValueError("Tỷ lệ train/test không hợp lệ.")

    X = training_df[FEATURE_COLUMNS]
    y = training_df[TARGET_COLUMN]
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = make_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions = pd.DataFrame(
        {
            "date": training_df["date"].iloc[split_index:].values,
            "actual_close": y_test.values,
            "predicted_close": y_pred,
            "error": y_test.values - y_pred,
        }
    )
    predictions["error_pct"] = np.where(
        predictions["actual_close"] != 0,
        predictions["error"] / predictions["actual_close"] * 100,
        np.nan,
    )

    metrics = {
        "Mô hình": model_name,
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAPE": mape(y_test, y_pred),
        "R2": float(r2_score(y_test, y_pred)),
        "Train rows": int(len(X_train)),
        "Test rows": int(len(X_test)),
    }

    next_prediction = predict_next_close(model, feature_df)
    return TrainingResult(
        model_name=model_name,
        model=model,
        metrics=metrics,
        predictions=predictions,
        next_prediction=next_prediction,
    )


def predict_next_close(model, feature_df: pd.DataFrame) -> dict[str, float | str]:
    if feature_df.empty:
        raise ValueError("Không có dữ liệu đặc trưng để dự báo.")

    latest = feature_df.sort_values("date").iloc[-1]
    latest_features = latest[FEATURE_COLUMNS].to_frame().T
    predicted_close = float(model.predict(latest_features)[0])
    last_close = float(latest["close"])
    change = predicted_close - last_close
    change_pct = change / last_close * 100 if last_close != 0 else np.nan

    if change > 0:
        trend = "Tăng"
    elif change < 0:
        trend = "Giảm"
    else:
        trend = "Không đổi"

    return {
        "last_date": pd.Timestamp(latest["date"]).date().isoformat(),
        "forecast_date": next_business_day(pd.Timestamp(latest["date"])).date().isoformat(),
        "last_close": last_close,
        "predicted_close": predicted_close,
        "change": change,
        "change_pct": change_pct,
        "trend": trend,
    }
