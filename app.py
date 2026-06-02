from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_sync import sync_vnstock_data
from features import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    INDUSTRY_GROUPS,
    TARGET_COLUMN,
    create_features,
    feature_description_table,
    filter_industry,
    filter_symbol,
    load_stock_data,
    normalize_sector_prices,
    validate_dataset,
)
from modeling import MODEL_OPTIONS, train_and_predict


APP_TITLE = "Dự báo giá cổ phiếu theo nhóm ngành"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="VN",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f7fb;
            --panel: #ffffff;
            --text: #17202a;
            --muted: #667085;
            --line: #d9e0ea;
            --accent: #2563eb;
            --accent-soft: #e8f0ff;
            --green: #16803c;
            --red: #c62828;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .main .block-container {
            padding-top: 1.25rem;
            max-width: 1320px;
        }

        section[data-testid="stSidebar"] {
            background: #111827;
        }

        section[data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .app-header {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 16px;
        }

        .app-title {
            font-size: 28px;
            font-weight: 760;
            margin-bottom: 4px;
            color: var(--text);
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 14px;
        }

        .notice {
            background: #fff8e1;
            border: 1px solid #f1d37a;
            border-radius: 8px;
            padding: 12px 14px;
            color: #6d4c00;
            margin: 12px 0 16px 0;
        }

        .result-box {
            background: var(--accent-soft);
            border: 1px solid #b9cdfc;
            border-radius: 8px;
            padding: 14px 16px;
            margin: 12px 0;
        }

        .result-title {
            color: var(--accent);
            font-weight: 750;
            margin-bottom: 6px;
        }

        .small-muted {
            color: var(--muted);
            font-size: 13px;
        }

        div.stButton > button {
            border-radius: 8px;
            font-weight: 700;
            background: var(--accent);
            color: white;
            border: 0;
            min-height: 42px;
        }

        div.stButton > button:hover {
            background: #1d4ed8;
            color: white;
            border: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_stock_data(DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def build_features(data: pd.DataFrame) -> pd.DataFrame:
    return create_features(data)


def reset_data_cache() -> None:
    load_data.clear()
    build_features.clear()
    st.session_state.pop("training_result", None)
    st.session_state.pop("training_key", None)


def fmt_price(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}"


def fmt_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f}"


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}%"


def header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">Dự báo giá cổ phiếu Việt Nam theo nhóm ngành</div>
            <div class="app-subtitle">
                Ứng dụng Streamlit dùng dữ liệu lịch sử OHLCV, tạo đặc trưng kỹ thuật,
                huấn luyện mô hình học máy và dự báo giá đóng cửa phiên kế tiếp.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_note() -> None:
    st.markdown(
        """
        <div class="notice">
            Kết quả dự báo chỉ phục vụ mục đích học tập và trình bày đồ án.
            Đây không phải khuyến nghị mua, bán hoặc nắm giữ chứng khoán.
        </div>
        """,
        unsafe_allow_html=True,
    )


def sync_data_panel(raw_df: pd.DataFrame, current_symbol: str) -> None:
    with st.sidebar.expander("Đồng bộ dữ liệu vnstock"):
        st.caption("Cập nhật dữ liệu OHLCV và lưu lại vào data/data.csv.")
        scope = st.radio(
            "Phạm vi cập nhật",
            ["Mã đang chọn", "Tất cả mã trong app"],
            label_visibility="visible",
        )

        latest_date = raw_df["date"].max().date()
        today = date.today()
        default_start = latest_date if latest_date <= today else today

        start_date = st.date_input("Từ ngày", value=default_start)
        end_date = st.date_input("Đến ngày", value=today)

        if scope == "Mã đang chọn":
            symbols_to_sync = [current_symbol]
        else:
            symbols_to_sync = [symbol for symbols in INDUSTRY_GROUPS.values() for symbol in symbols]

        st.caption("Mã sẽ cập nhật: " + ", ".join(symbols_to_sync))

        if st.button("Đồng bộ dữ liệu từ vnstock", use_container_width=True):
            try:
                with st.spinner("Đang lấy dữ liệu từ vnstock..."):
                    result = sync_vnstock_data(symbols_to_sync, start_date, end_date, DEFAULT_DATA_PATH)
                reset_data_cache()
                st.success(
                    "Đã đồng bộ "
                    f"{result['downloaded_rows']} dòng cho {', '.join(result['synced_symbols'])}. "
                    f"Tổng dữ liệu hiện có: {result['total_rows']} dòng."
                )
                if result["errors"]:
                    messages = [f"{symbol}: {message}" for symbol, message in result["errors"].items()]
                    st.warning("Một số mã có cảnh báo: " + " | ".join(messages))
                st.rerun()
            except Exception as exc:
                st.error(f"Không thể đồng bộ dữ liệu: {exc}")


def price_line_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="lines",
            name="Giá đóng cửa",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis_title="Ngày",
        yaxis_title="Giá đóng cửa",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )
    return fig


def candlestick_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
                increasing_line_color="#16803c",
                decreasing_line_color="#c62828",
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis_title="Ngày",
        yaxis_title="Giá",
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_rangeslider_visible=False,
    )
    return fig


def sector_comparison_chart(df: pd.DataFrame, title: str) -> go.Figure:
    normalized = normalize_sector_prices(df)
    fig = go.Figure()
    for symbol, group in normalized.groupby("symbol"):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["normalized_close"],
                mode="lines",
                name=symbol,
            )
        )
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis_title="Ngày",
        yaxis_title="Giá chuẩn hóa",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )
    return fig


def prediction_chart(predictions: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["actual_close"],
            mode="lines",
            name="Giá thực tế",
            line=dict(color="#111827", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["predicted_close"],
            mode="lines",
            name="Giá dự báo",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis_title="Ngày",
        yaxis_title="Giá đóng cửa",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )
    return fig


def show_overview(raw_df: pd.DataFrame, sector_df: pd.DataFrame, symbol_df: pd.DataFrame, symbol: str) -> None:
    latest = symbol_df.iloc[-1]
    first = symbol_df.iloc[0]
    total_return = (latest["close"] / first["close"] - 1) * 100 if first["close"] else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mã cổ phiếu", symbol)
    col2.metric("Giá mới nhất", fmt_price(latest["close"]))
    col3.metric("Số phiên dữ liệu", fmt_int(len(symbol_df)))
    col4.metric("Tăng/giảm toàn kỳ", fmt_pct(total_return))

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.plotly_chart(
            price_line_chart(symbol_df, f"Diễn biến giá đóng cửa - {symbol}"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            sector_comparison_chart(sector_df, "So sánh cổ phiếu cùng nhóm ngành"),
            use_container_width=True,
        )

    st.plotly_chart(
        candlestick_chart(symbol_df.tail(180), f"Biểu đồ nến 180 phiên gần nhất - {symbol}"),
        use_container_width=True,
    )

    st.subheader("Bảng kiểm tra dữ liệu")
    st.dataframe(validate_dataset(raw_df), use_container_width=True, hide_index=True)

    st.subheader(f"Dữ liệu gần nhất của {symbol}")
    st.dataframe(symbol_df.tail(120), use_container_width=True, hide_index=True)


def show_features(feature_df: pd.DataFrame, symbol_features: pd.DataFrame, symbol: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Số đặc trưng", fmt_int(len(FEATURE_COLUMNS)))
    col2.metric("Dòng đặc trưng của mã", fmt_int(len(symbol_features)))
    col3.metric("Biến mục tiêu", TARGET_COLUMN)
    col4.metric("Dòng train hợp lệ", fmt_int(symbol_features[TARGET_COLUMN].notna().sum()))

    st.subheader("Danh sách đặc trưng")
    st.dataframe(feature_description_table(), use_container_width=True, hide_index=True)

    st.subheader(f"Bảng đặc trưng gần nhất - {symbol}")
    display_columns = ["date", "symbol", *FEATURE_COLUMNS, TARGET_COLUMN, "target_up"]
    st.dataframe(symbol_features[display_columns].tail(120), use_container_width=True, hide_index=True)

    st.subheader("Dữ liệu đặc trưng toàn bộ các mã")
    st.dataframe(
        feature_df[["date", "symbol", "close", "ma5", "ma20", "ma50", TARGET_COLUMN, "target_up"]].tail(200),
        use_container_width=True,
        hide_index=True,
    )


def show_modeling(symbol_features: pd.DataFrame, model_name: str, test_size: float, symbol: str) -> None:
    train_count = symbol_features[TARGET_COLUMN].notna().sum()
    st.write(f"Dữ liệu huấn luyện khả dụng cho {symbol}: {train_count} dòng.")

    if st.button("Huấn luyện mô hình và dự báo", use_container_width=True):
        with st.spinner("Đang huấn luyện mô hình..."):
            st.session_state["training_result"] = train_and_predict(symbol_features, model_name, test_size)
            st.session_state["training_key"] = (symbol, model_name, test_size)

    result = st.session_state.get("training_result")
    result_key = st.session_state.get("training_key")
    current_key = (symbol, model_name, test_size)

    if result is None or result_key != current_key:
        st.info("Bấm nút huấn luyện để xem kết quả đánh giá và dự báo.")
        return

    metrics = result.metrics
    next_prediction = result.next_prediction
    predictions = result.predictions

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", fmt_price(metrics["MAE"]))
    col2.metric("RMSE", fmt_price(metrics["RMSE"]))
    col3.metric("MAPE", fmt_pct(metrics["MAPE"]))
    col4.metric("R2", f"{metrics['R2']:.4f}")

    trend_delta = fmt_pct(next_prediction["change_pct"])
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">Kết quả dự báo phiên kế tiếp</div>
            Mô hình <b>{result.model_name}</b> dự báo giá đóng cửa của <b>{symbol}</b>
            vào ngày giao dịch kế tiếp <b>{next_prediction["forecast_date"]}</b> là
            <b>{fmt_price(next_prediction["predicted_close"])}</b>.
            Giá đóng cửa gần nhất ngày <b>{next_prediction["last_date"]}</b> là
            <b>{fmt_price(next_prediction["last_close"])}</b>.
            Xu hướng dự báo: <b>{next_prediction["trend"]}</b>
            ({fmt_price(next_prediction["change"])}; {trend_delta}).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        prediction_chart(predictions, f"So sánh giá thực tế và dự báo - {symbol}"),
        use_container_width=True,
    )

    st.subheader("Bảng kết quả trên tập kiểm thử")
    st.dataframe(predictions.tail(160), use_container_width=True, hide_index=True)

    csv_data = predictions.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Tải kết quả dự báo CSV",
        data=csv_data,
        file_name=f"du_bao_{symbol}_{model_name.replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def show_report(industry: str, symbol: str, model_name: str) -> None:
    st.subheader("Gợi ý trình bày đồ án")
    st.markdown(
        f"""
        **Tên đề tài:** Ứng dụng dự báo giá cổ phiếu Việt Nam theo nhóm ngành bằng mô hình học máy.

        **Phạm vi demo:** Nhóm ngành đang chọn là **{industry}**, mã cổ phiếu đang phân tích là **{symbol}**.

        **Quy trình thực hiện:**
        1. Thu thập dữ liệu lịch sử gồm ngày, giá mở cửa, giá cao nhất, giá thấp nhất, giá đóng cửa và khối lượng.
        2. Làm sạch dữ liệu, sắp xếp theo thời gian và tách theo mã cổ phiếu.
        3. Tạo các đặc trưng kỹ thuật như return, moving average, tỷ lệ giá so với MA, biến trễ và độ biến động.
        4. Chia dữ liệu train/test theo thứ tự thời gian.
        5. Huấn luyện mô hình **{model_name}** và đánh giá bằng MAE, RMSE, MAPE, R2.
        6. Dự báo giá đóng cửa của phiên giao dịch kế tiếp.

        **Ý nghĩa:** Ứng dụng giúp người dùng xem dữ liệu, so sánh cổ phiếu cùng nhóm ngành,
        chạy mô hình dự báo cơ bản và hiểu các bước xây dựng một hệ thống phân tích dữ liệu tài chính.
        """
    )


def main() -> None:
    inject_css()
    header()
    warning_note()

    try:
        raw_df = load_data()
        feature_df = build_features(raw_df)
    except Exception as exc:
        st.error(f"Không thể khởi tạo ứng dụng: {exc}")
        st.stop()

    with st.sidebar:
        st.title("Cấu hình")
        industry = st.selectbox("Nhóm ngành", list(INDUSTRY_GROUPS.keys()))
        available_symbols = INDUSTRY_GROUPS[industry]
        symbol = st.selectbox("Mã cổ phiếu", available_symbols)
        model_name = st.selectbox("Mô hình dự báo", MODEL_OPTIONS)
        test_size = st.slider("Tỷ lệ dữ liệu test", 0.10, 0.40, 0.20, 0.05)

        sync_data_panel(raw_df, symbol)

        st.divider()
        st.caption("Dữ liệu mặc định")
        st.write(str(DEFAULT_DATA_PATH.name))

    sector_df = filter_industry(raw_df, industry)
    symbol_df = filter_symbol(raw_df, symbol)
    symbol_features = filter_symbol(feature_df, symbol)

    if symbol_df.empty or symbol_features.empty:
        st.error("Không đủ dữ liệu cho mã cổ phiếu đã chọn.")
        st.stop()

    tab_overview, tab_features, tab_model, tab_report = st.tabs(
        ["Tổng quan", "Đặc trưng", "Huấn luyện và dự báo", "Báo cáo đồ án"]
    )

    with tab_overview:
        show_overview(raw_df, sector_df, symbol_df, symbol)

    with tab_features:
        show_features(feature_df, symbol_features, symbol)

    with tab_model:
        show_modeling(symbol_features, model_name, test_size, symbol)

    with tab_report:
        show_report(industry, symbol, model_name)


if __name__ == "__main__":
    main()
