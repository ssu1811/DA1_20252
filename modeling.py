from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from features import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_RETURN_COLUMN


MODEL_OPTIONS = ["Linear Regression", "Random Forest", "SVR", "LSTM"]
RETURN_TARGET_MODELS = {"Random Forest", "SVR", "LSTM"}
LSTM_LOOKBACK = 20
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 32
LSTM_RETURN_CLIP = 0.10


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
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=5,
                        min_samples_split=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if model_name == "SVR":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf", C=1, gamma="scale", epsilon=0.005)),
                ]
            ),
            transformer=StandardScaler(),
        )
    raise ValueError(f"Mô hình chưa được hỗ trợ: {model_name}")


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100)


def metric_dict(model_name: str, y_true, y_pred, train_rows: int, test_rows: int) -> dict[str, float | int | str]:
    mse_value = float(mean_squared_error(y_true, y_pred))
    return {
        "Mô hình": model_name,
        "MSE": mse_value,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse_value)),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Train rows": int(train_rows),
        "Test rows": int(test_rows),
    }


def next_business_day(value: pd.Timestamp) -> pd.Timestamp:
    next_day = pd.Timestamp(value).normalize() + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def train_and_predict(feature_df: pd.DataFrame, model_name: str, test_size: float) -> TrainingResult:
    if model_name == "LSTM":
        return train_lstm(feature_df.copy(), test_size)

    target_column = TARGET_RETURN_COLUMN if model_name in RETURN_TARGET_MODELS else TARGET_COLUMN
    training_df = feature_df.dropna(subset=[TARGET_COLUMN, target_column]).copy()
    if len(training_df) < 80:
        raise ValueError("Cần ít nhất 80 dòng dữ liệu hợp lệ sau khi tạo đặc trưng.")

    split_index = int(len(training_df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(training_df):
        raise ValueError("Tỷ lệ train/test không hợp lệ.")

    X = training_df[FEATURE_COLUMNS]
    y = training_df[target_column]
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    actual_close_test = training_df[TARGET_COLUMN].iloc[split_index:]

    model = make_model(model_name)
    model.fit(X_train, y_train)
    raw_pred = model.predict(X_test)
    if model_name in RETURN_TARGET_MODELS:
        y_pred = training_df["close"].iloc[split_index:].values * (1 + raw_pred)
    else:
        y_pred = raw_pred

    prediction_dates = (
        training_df["target_date"].iloc[split_index:].values
        if "target_date" in training_df.columns
        else training_df["date"].iloc[split_index:].values
    )
    predictions = prediction_frame(prediction_dates, actual_close_test.values, y_pred)
    metrics = metric_dict(model_name, actual_close_test, y_pred, len(X_train), len(X_test))

    next_prediction = predict_next_close(model, feature_df, model_name)
    return TrainingResult(
        model_name=model_name,
        model=model,
        metrics=metrics,
        predictions=predictions,
        next_prediction=next_prediction,
    )


def prediction_frame(dates, actual_close, predicted_close) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "date": dates,
            "actual_close": actual_close,
            "predicted_close": predicted_close,
        }
    )
    predictions["error"] = predictions["actual_close"] - predictions["predicted_close"]
    predictions["error_pct"] = np.where(
        predictions["actual_close"] != 0,
        predictions["error"] / predictions["actual_close"] * 100,
        np.nan,
    )
    return predictions


def train_lstm(feature_df: pd.DataFrame, test_size: float) -> TrainingResult:
    try:
        tf = import_module("tensorflow")
        callbacks_module = import_module("tensorflow.keras.callbacks")
        layers_module = import_module("tensorflow.keras.layers")
        models_module = import_module("tensorflow.keras.models")
        optimizers_module = import_module("tensorflow.keras.optimizers")

        EarlyStopping = callbacks_module.EarlyStopping
        LSTM = layers_module.LSTM
        Dense = layers_module.Dense
        Dropout = layers_module.Dropout
        Input = layers_module.Input
        Sequential = models_module.Sequential
        Adam = optimizers_module.Adam
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Chưa cài TensorFlow nên chưa chạy được mô hình LSTM. "
            "App vẫn chạy được các mô hình Linear Regression, Random Forest và SVR. "
            "Nếu muốn dùng LSTM, hãy cài thêm bằng lệnh: pip install -r requirements-lstm.txt"
        ) from exc

    training_df = feature_df.dropna(subset=[TARGET_COLUMN, TARGET_RETURN_COLUMN]).copy()
    if len(training_df) < 80:
        raise ValueError("Cần ít nhất 80 dòng dữ liệu hợp lệ sau khi tạo đặc trưng.")

    sequence_count = len(training_df) - LSTM_LOOKBACK + 1
    if sequence_count < 40:
        raise ValueError(f"LSTM cần ít nhất khoảng {LSTM_LOOKBACK + 40} dòng dữ liệu sau khi tạo đặc trưng.")

    split_index = int(sequence_count * (1 - test_size))
    if split_index <= 0 or split_index >= sequence_count:
        raise ValueError("Tỷ lệ train/test không hợp lệ cho LSTM.")

    train_raw_end = LSTM_LOOKBACK + split_index - 1
    scaler = StandardScaler()
    scaler.fit(training_df[FEATURE_COLUMNS].iloc[:train_raw_end])
    scaled_features = scaler.transform(training_df[FEATURE_COLUMNS])
    target_scaler = StandardScaler()

    X_seq = []
    y_return_seq = []
    actual_close_seq = []
    base_close_seq = []
    date_seq = []
    for current_index in range(LSTM_LOOKBACK - 1, len(training_df)):
        start_index = current_index - LSTM_LOOKBACK + 1
        X_seq.append(scaled_features[start_index : current_index + 1])
        y_return_seq.append(training_df[TARGET_RETURN_COLUMN].iloc[current_index])
        actual_close_seq.append(training_df[TARGET_COLUMN].iloc[current_index])
        base_close_seq.append(training_df["close"].iloc[current_index])
        if "target_date" in training_df.columns:
            date_seq.append(training_df["target_date"].iloc[current_index])
        else:
            date_seq.append(training_df["date"].iloc[current_index])

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_return_seq = np.asarray(y_return_seq, dtype=np.float32)
    actual_close_seq = np.asarray(actual_close_seq, dtype=float)
    base_close_seq = np.asarray(base_close_seq, dtype=float)
    date_seq = np.asarray(date_seq)

    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train = y_return_seq[:split_index]
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    actual_close_test = actual_close_seq[split_index:]
    base_close_test = base_close_seq[split_index:]

    np.random.seed(42)
    tf.random.set_seed(42)
    model = Sequential(
        [
            Input(shape=(LSTM_LOOKBACK, len(FEATURE_COLUMNS))),
            LSTM(48, dropout=0.1),
            Dropout(0.15),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(
        X_train,
        y_train_scaled,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_split=0.15,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )

    predicted_return_scaled = model.predict(X_test, verbose=0).reshape(-1)
    predicted_return = target_scaler.inverse_transform(predicted_return_scaled.reshape(-1, 1)).reshape(-1)
    predicted_return = np.clip(predicted_return, -LSTM_RETURN_CLIP, LSTM_RETURN_CLIP)
    predicted_close = base_close_test * (1 + predicted_return)
    predictions = prediction_frame(date_seq[split_index:], actual_close_test, predicted_close)

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "return_clip": LSTM_RETURN_CLIP,
        "lookback": LSTM_LOOKBACK,
        "feature_columns": FEATURE_COLUMNS,
    }
    next_prediction = predict_next_close(model_bundle, feature_df, "LSTM")
    metrics = metric_dict("LSTM", actual_close_test, predicted_close, len(X_train), len(X_test))

    return TrainingResult(
        model_name="LSTM",
        model=model_bundle,
        metrics=metrics,
        predictions=predictions,
        next_prediction=next_prediction,
    )


def predict_next_close(model, feature_df: pd.DataFrame, model_name: str | None = None) -> dict[str, float | str]:
    if feature_df.empty:
        raise ValueError("Không có dữ liệu đặc trưng để dự báo.")

    if model_name == "LSTM":
        return predict_next_close_lstm(model, feature_df)

    latest = feature_df.sort_values("date").iloc[-1]
    latest_features = latest[FEATURE_COLUMNS].to_frame().T
    last_close = float(latest["close"])
    raw_prediction = float(model.predict(latest_features)[0])
    if model_name in RETURN_TARGET_MODELS:
        predicted_close = last_close * (1 + raw_prediction)
    else:
        predicted_close = raw_prediction

    return next_prediction_dict(latest["date"], last_close, predicted_close)


def predict_next_close_lstm(model_bundle: dict[str, object], feature_df: pd.DataFrame) -> dict[str, float | str]:
    clean_df = feature_df.dropna(subset=FEATURE_COLUMNS).sort_values("date")
    lookback = int(model_bundle["lookback"])
    if len(clean_df) < lookback:
        raise ValueError(f"LSTM cần ít nhất {lookback} phiên dữ liệu đặc trưng để dự báo phiên kế tiếp.")

    scaler = model_bundle["scaler"]
    target_scaler = model_bundle["target_scaler"]
    model = model_bundle["model"]
    latest_rows = clean_df.tail(lookback)
    latest_features = scaler.transform(latest_rows[FEATURE_COLUMNS])
    latest_sequence = latest_features.reshape(1, lookback, len(FEATURE_COLUMNS))
    predicted_return_scaled = float(model.predict(latest_sequence, verbose=0).reshape(-1)[0])
    predicted_return = float(target_scaler.inverse_transform([[predicted_return_scaled]])[0][0])
    predicted_return = float(np.clip(predicted_return, -float(model_bundle["return_clip"]), float(model_bundle["return_clip"])))

    latest = latest_rows.iloc[-1]
    last_close = float(latest["close"])
    predicted_close = last_close * (1 + predicted_return)
    return next_prediction_dict(latest["date"], last_close, predicted_close)


def next_prediction_dict(latest_date, last_close: float, predicted_close: float) -> dict[str, float | str]:
    change = predicted_close - last_close
    change_pct = change / last_close * 100 if last_close != 0 else np.nan

    if change > 0:
        trend = "Tăng"
    elif change < 0:
        trend = "Giảm"
    else:
        trend = "Không đổi"

    latest_timestamp = pd.Timestamp(latest_date)
    return {
        "last_date": latest_timestamp.date().isoformat(),
        "forecast_date": next_business_day(latest_timestamp).date().isoformat(),
        "last_close": last_close,
        "predicted_close": predicted_close,
        "change": change,
        "change_pct": change_pct,
        "trend": trend,
    }
