from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "data.csv"

INDUSTRY_GROUPS: dict[str, list[str]] = {
    "Ngân hàng": ["ACB", "BID", "VCB"],
    "Công nghệ": ["FPT", "CMG", "ELC"],
    "Tiêu dùng": ["VNM", "SAB", "MSN"],
}

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "symbol"]

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_change",
    "price_change_pct",
    "high_low_pct",
    "return",
    "ma5",
    "ma10",
    "ma20",
    "ma50",
    "volume_ma20",
    "close_ma5_ratio",
    "close_ma20_ratio",
    "close_ma50_ratio",
    "close_lag1",
    "close_lag2",
    "close_lag3",
    "volume_lag1",
    "volatility_10",
]

TARGET_COLUMN = "target_next_close"
TARGET_RETURN_COLUMN = "target_return"

FEATURE_DESCRIPTIONS = {
    "open": "Giá mở cửa của phiên giao dịch.",
    "high": "Giá cao nhất trong phiên.",
    "low": "Giá thấp nhất trong phiên.",
    "close": "Giá đóng cửa của phiên.",
    "volume": "Khối lượng giao dịch.",
    "price_change": "Chênh lệch giữa giá đóng cửa và mở cửa.",
    "price_change_pct": "Tỷ lệ biến động trong phiên so với giá mở cửa.",
    "high_low_pct": "Biên độ dao động trong phiên so với giá đóng cửa.",
    "return": "Tỷ suất sinh lời hằng ngày của giá đóng cửa.",
    "ma5": "Trung bình động 5 phiên.",
    "ma10": "Trung bình động 10 phiên.",
    "ma20": "Trung bình động 20 phiên.",
    "ma50": "Trung bình động 50 phiên.",
    "volume_ma20": "Khối lượng giao dịch trung bình 20 phiên.",
    "close_ma5_ratio": "Tỷ lệ giá đóng cửa so với MA5.",
    "close_ma20_ratio": "Tỷ lệ giá đóng cửa so với MA20.",
    "close_ma50_ratio": "Tỷ lệ giá đóng cửa so với MA50.",
    "close_lag1": "Giá đóng cửa của phiên trước.",
    "close_lag2": "Giá đóng cửa cách 2 phiên.",
    "close_lag3": "Giá đóng cửa cách 3 phiên.",
    "volume_lag1": "Khối lượng giao dịch của phiên trước.",
    "volatility_10": "Độ biến động của return trong 10 phiên gần nhất.",
}


def load_stock_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read and clean the stock OHLCV dataset used by the Streamlit app."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.drop_duplicates(subset=["symbol", "date"])
    df = df[df["volume"] >= 0]
    df = df[
        (df["low"] <= df["high"])
        & (df["open"].between(df["low"], df["high"]))
        & (df["close"].between(df["low"], df["high"]))
    ]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def get_symbols_by_industry(industry: str) -> list[str]:
    return INDUSTRY_GROUPS.get(industry, [])


def filter_industry(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    symbols = get_symbols_by_industry(industry)
    return df[df["symbol"].isin(symbols)].copy()


def filter_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact quality report for the current dataset."""
    rows = []
    for symbol, group in df.groupby("symbol"):
        rows.append(
            {
                "Mã": symbol,
                "Số phiên": len(group),
                "Từ ngày": group["date"].min().date(),
                "Đến ngày": group["date"].max().date(),
                "Giá mới nhất": group.sort_values("date")["close"].iloc[-1],
                "Volume TB": group["volume"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("Mã").reset_index(drop=True)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical features and the next-close target for each symbol.

    The latest row is kept even when target_next_close is missing, because the
    app uses that row to forecast the next trading session.
    """
    if df.empty:
        return pd.DataFrame()

    feature_df = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped_close = feature_df.groupby("symbol")["close"]
    grouped_volume = feature_df.groupby("symbol")["volume"]

    feature_df["price_change"] = feature_df["close"] - feature_df["open"]
    feature_df["price_change_pct"] = feature_df["price_change"] / feature_df["open"]
    feature_df["high_low_pct"] = (feature_df["high"] - feature_df["low"]) / feature_df["close"]
    feature_df["return"] = grouped_close.pct_change()

    feature_df["ma5"] = grouped_close.transform(lambda values: values.rolling(window=5).mean())
    feature_df["ma10"] = grouped_close.transform(lambda values: values.rolling(window=10).mean())
    feature_df["ma20"] = grouped_close.transform(lambda values: values.rolling(window=20).mean())
    feature_df["ma50"] = grouped_close.transform(lambda values: values.rolling(window=50).mean())
    feature_df["volume_ma20"] = grouped_volume.transform(lambda values: values.rolling(window=20).mean())

    feature_df["close_ma5_ratio"] = feature_df["close"] / feature_df["ma5"]
    feature_df["close_ma20_ratio"] = feature_df["close"] / feature_df["ma20"]
    feature_df["close_ma50_ratio"] = feature_df["close"] / feature_df["ma50"]

    feature_df["close_lag1"] = grouped_close.shift(1)
    feature_df["close_lag2"] = grouped_close.shift(2)
    feature_df["close_lag3"] = grouped_close.shift(3)
    feature_df["volume_lag1"] = grouped_volume.shift(1)

    feature_df["volatility_10"] = feature_df.groupby("symbol")["return"].transform(
        lambda values: values.rolling(window=10).std()
    )

    feature_df[TARGET_COLUMN] = grouped_close.shift(-1)
    feature_df["target_date"] = feature_df.groupby("symbol")["date"].shift(-1)
    feature_df[TARGET_RETURN_COLUMN] = feature_df[TARGET_COLUMN] / feature_df["close"] - 1
    feature_df["target_up"] = np.where(
        feature_df[TARGET_COLUMN].isna(),
        np.nan,
        (feature_df[TARGET_COLUMN] > feature_df["close"]).astype(int),
    )

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.dropna(subset=FEATURE_COLUMNS)
    return feature_df.reset_index(drop=True)


def feature_description_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"Đặc trưng": name, "Ý nghĩa": description} for name, description in FEATURE_DESCRIPTIONS.items()]
    )


def normalize_sector_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each symbol's close price to 100 at the first available date."""
    rows = []
    for symbol, group in df.sort_values("date").groupby("symbol"):
        if group.empty:
            continue
        base = group["close"].iloc[0]
        if base == 0:
            continue
        temp = group[["date", "symbol", "close"]].copy()
        temp["normalized_close"] = temp["close"] / base * 100
        rows.append(temp)

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "close", "normalized_close"])
    return pd.concat(rows, ignore_index=True)
