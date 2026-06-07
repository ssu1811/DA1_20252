from __future__ import annotations

from datetime import date
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from features import DEFAULT_DATA_PATH, REQUIRED_COLUMNS, load_stock_data

DEFAULT_FETCH_COUNT = 2500
DEAD_PROXY_MARKERS = ("127.0.0.1:9", "localhost:9")
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _date_to_text(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _remove_dead_proxy_env() -> dict[str, str]:
    """Remove sandbox-style proxy values that intentionally block internet access."""
    removed = {}
    for key in PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if value and any(marker in value for marker in DEAD_PROXY_MARKERS):
            removed[key] = value
            os.environ.pop(key, None)
    return removed


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    lower_message = message.lower()
    if "127.0.0.1:9" in message or "proxyerror" in lower_message:
        return (
            "Không kết nối được tới vnstock vì tiến trình đang dùng proxy chặn mạng "
            "(127.0.0.1:9). Hãy chạy app bằng terminal bình thường hoặc để app tự bỏ proxy chặn."
        )
    if "retryerror" in lower_message or "connectionerror" in lower_message:
        return (
            "Không kết nối được tới API vnstock sau nhiều lần thử. "
            "Hãy kiểm tra mạng, VPN/proxy/firewall hoặc thử lại với khoảng ngày ngắn hơn."
        )
    return message


def _market_client():
    """Create a vnstock Market client with support for recent and older imports."""
    _remove_dead_proxy_env()
    try:
        from vnstock.ui import Market
    except ImportError:
        try:
            from vnstock import Market
        except ImportError as exc:
            raise ImportError(
                "Chưa cài thư viện vnstock. Hãy chạy: pip install -r requirements.txt"
            ) from exc
    return Market()


def _normalize_vnstock_frame(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = raw_df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})

    aliases = {
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
        "vol": "volume",
    }
    df = df.rename(columns={old: new for old, new in aliases.items() if old in df.columns})

    missing = [column for column in ["date", "open", "high", "low", "close", "volume"] if column not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu vnstock của {symbol} thiếu cột: {', '.join(missing)}")

    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["symbol"] = symbol.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["volume"] >= 0]
    df = df.sort_values("date").drop_duplicates(subset=["symbol", "date"])
    return df[REQUIRED_COLUMNS].reset_index(drop=True)


def fetch_symbol_ohlcv(
    symbol: str,
    start_date: date | str,
    end_date: date | str,
    count: int = DEFAULT_FETCH_COUNT,
) -> pd.DataFrame:
    _remove_dead_proxy_env()
    market = _market_client()
    raw_df = market.equity(symbol.upper()).ohlcv(
        start=_date_to_text(start_date),
        end=_date_to_text(end_date),
        interval="1D",
        count=count,
    )
    return _normalize_vnstock_frame(raw_df, symbol)


def sync_vnstock_data(
    symbols: Iterable[str],
    start_date: date | str,
    end_date: date | str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    count: int = DEFAULT_FETCH_COUNT,
) -> dict[str, object]:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if start_ts > end_ts:
        raise ValueError("Ngày bắt đầu không được lớn hơn ngày kết thúc.")

    data_path = Path(data_path)
    unique_symbols = list(dict.fromkeys(symbol.upper().strip() for symbol in symbols if symbol.strip()))
    if not unique_symbols:
        raise ValueError("Chưa có mã cổ phiếu để đồng bộ.")

    old_df = load_stock_data(data_path) if data_path.exists() else pd.DataFrame(columns=REQUIRED_COLUMNS)
    new_frames = []
    errors: dict[str, str] = {}

    for symbol in unique_symbols:
        try:
            symbol_df = fetch_symbol_ohlcv(symbol, start_ts.date(), end_ts.date(), count=count)
            if symbol_df.empty:
                errors[symbol] = "Không có dữ liệu mới trong khoảng ngày đã chọn."
            else:
                new_frames.append(symbol_df)
        except Exception as exc:
            errors[symbol] = _friendly_error(exc)

    if not new_frames:
        detail = "; ".join(f"{symbol}: {message}" for symbol, message in errors.items())
        raise RuntimeError(f"Không đồng bộ được dữ liệu từ vnstock. {detail}")

    new_df = pd.concat(new_frames, ignore_index=True)
    if old_df.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat([old_df, new_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=REQUIRED_COLUMNS)
    combined = combined.sort_values(["symbol", "date"])
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    combined = combined[REQUIRED_COLUMNS].reset_index(drop=True)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = combined.copy()
    output_df["date"] = output_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_df.to_csv(data_path, index=False, encoding="utf-8-sig")

    return {
        "requested_symbols": unique_symbols,
        "synced_symbols": sorted(new_df["symbol"].unique().tolist()),
        "downloaded_rows": int(len(new_df)),
        "total_rows": int(len(combined)),
        "from_date": combined["date"].min().date().isoformat(),
        "to_date": combined["date"].max().date().isoformat(),
        "errors": errors,
    }
