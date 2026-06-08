from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
TRAINING_PIPELINE_VERSION = "return-target-lstm-svr-rf-v5"


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
            --notice-bg: #fff8e1;
            --notice-border: #f1d37a;
            --notice-text: #6d4c00;
            --result-border: #b9cdfc;
            --error-good-bg: #dcfce7;
            --error-good-text: #166534;
            --error-warn-bg: #fef3c7;
            --error-warn-text: #92400e;
            --error-bad-bg: #fee2e2;
            --error-bad-text: #991b1b;
        }

        html[data-app-theme="dark"],
        html[data-theme="dark"],
        body[data-theme="dark"],
        .stApp[data-theme="dark"],
        [data-testid="stAppViewContainer"][data-theme="dark"],
        [data-baseweb-theme="dark"] {
            --bg: #0f172a;
            --panel: #111827;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --line: #334155;
            --accent: #60a5fa;
            --accent-soft: #172554;
            --green: #86efac;
            --red: #fca5a5;
            --notice-bg: #3a2f12;
            --notice-border: #a16207;
            --notice-text: #fde68a;
            --result-border: #1d4ed8;
            --error-good-bg: #123524;
            --error-good-text: #86efac;
            --error-warn-bg: #3a2f12;
            --error-warn-text: #fde68a;
            --error-bad-bg: #3b1717;
            --error-bad-text: #fca5a5;
        }

        .stApp {
            background: var(--bg) !important;
            color: var(--text) !important;
        }

        .stApp main,
        .stApp main p,
        .stApp main span,
        .stApp main label,
        .stApp main div,
        .stApp main h1,
        .stApp main h2,
        .stApp main h3,
        .stApp main h4 {
            color: var(--text);
        }

        .main .block-container {
            padding-top: 1.25rem;
            max-width: 1320px;
        }

        section[data-testid="stSidebar"] {
            background: #111827;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }

        section[data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        .sidebar-menu-title {
            color: #ffffff;
            font-size: 34px;
            font-weight: 800;
            line-height: 0.1;
            margin: 0 0 30px 0;
            text-align: left;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"],
        section[data-testid="stSidebar"] div[data-baseweb="select"] *,
        section[data-testid="stSidebar"] div[data-baseweb="input"],
        section[data-testid="stSidebar"] div[data-baseweb="input"] *,
        section[data-testid="stSidebar"] input {
            color: #111827 !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
            background: #ffffff !important;
            border-color: #d1d5db !important;
        }

        section[data-testid="stSidebar"] div[role="listbox"] *,
        div[data-baseweb="popover"] * {
            color: #111827 !important;
        }

        html[data-app-theme="dark"] div[data-baseweb="popover"],
        html[data-app-theme="dark"] div[role="listbox"] {
            background: #111827 !important;
            border-color: #475569 !important;
        }

        html[data-app-theme="dark"] div[data-baseweb="popover"] *,
        html[data-app-theme="dark"] div[role="listbox"] *,
        html[data-app-theme="dark"] div[role="option"],
        html[data-app-theme="dark"] div[role="option"] * {
            color: #e5e7eb !important;
        }

        html[data-app-theme="dark"] div[role="option"]:hover,
        html[data-app-theme="dark"] div[role="option"][aria-selected="true"] {
            background: #1f2937 !important;
        }

        html[data-app-theme="dark"] div[role="option"]:hover *,
        html[data-app-theme="dark"] div[role="option"][aria-selected="true"] * {
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] details,
        section[data-testid="stSidebar"] details summary,
        section[data-testid="stSidebar"] details summary *,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary * {
            color: #111827 !important;
        }

        section[data-testid="stSidebar"] details summary,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
            background: #ffffff !important;
            border-radius: 8px !important;
        }

        section[data-testid="stSidebar"] details summary:hover,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover {
            background: #f3f4f6 !important;
        }

        section[data-testid="stSidebar"] details p,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] p,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] label,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] span {
            color: #d1d5db !important;
        }

        section[data-testid="stSidebar"] details summary,
        section[data-testid="stSidebar"] details summary *,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary *,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary span,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary svg {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
        }

        div[data-testid="stMetric"] *,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] *,
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] * {
            color: var(--text) !important;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-size: 0.88rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: clamp(1.7rem, 2.4vw, 3rem) !important;
            line-height: 1.12 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }

        .feature-summary-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 20px;
            min-height: 116px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .feature-summary-label {
            color: var(--muted) !important;
            font-size: 0.95rem;
            line-height: 1.2;
            margin-bottom: 12px;
        }

        .feature-summary-value {
            color: var(--text) !important;
            font-size: clamp(2rem, 2.8vw, 3rem);
            font-weight: 520;
            line-height: 1.08;
            letter-spacing: 0;
            white-space: normal;
            overflow-wrap: break-word;
        }

        .feature-summary-value.text-value {
            font-size: clamp(1.25rem, 1.65vw, 1.7rem);
            font-weight: 700;
            line-height: 1.22;
            white-space: normal;
            overflow-wrap: break-word;
        }

        div[data-testid="stTabs"] button[role="tab"] p {
            color: #334155 !important;
            font-weight: 650;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
            color: #ff4b4b !important;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            background: transparent !important;
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
            background: var(--notice-bg);
            border: 1px solid var(--notice-border);
            border-radius: 8px;
            padding: 12px 14px;
            color: var(--notice-text) !important;
            margin: 12px 0 16px 0;
        }

        .notice,
        .notice * {
            color: var(--notice-text) !important;
        }

        .result-box {
            background: var(--accent-soft);
            border: 1px solid var(--result-border);
            border-radius: 8px;
            padding: 14px 16px;
            margin: 12px 0;
            color: var(--text) !important;
        }

        .result-box,
        .result-box * {
            color: var(--text) !important;
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

        div.stButton > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius: 8px;
            font-weight: 700;
            background: var(--accent);
            color: #ffffff !important;
            border: 0;
            min-height: 42px;
        }

        div.stButton > button *,
        div[data-testid="stDownloadButton"] > button * {
            color: #ffffff !important;
        }

        div.stButton > button:hover,
        div[data-testid="stDownloadButton"] > button:hover,
        div.stButton > button:focus,
        div[data-testid="stDownloadButton"] > button:focus {
            background: #1d4ed8;
            color: #ffffff !important;
            border: 0;
        }

        .prediction-table-wrap {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: auto;
            max-height: 540px;
            background: var(--panel);
            margin-top: 8px;
            margin-bottom: 16px;
        }

        .prediction-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }

        .prediction-table th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: var(--panel);
            color: var(--muted) !important;
            text-align: left;
            font-weight: 700;
            padding: 11px 12px;
            border-bottom: 1px solid var(--line);
            border-right: 1px solid var(--line);
        }

        .prediction-table td {
            color: var(--text) !important;
            padding: 10px 12px;
            border-bottom: 1px solid var(--line);
            border-right: 1px solid var(--line);
            white-space: nowrap;
        }

        .prediction-table td.num {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        .prediction-table td.error-good,
        .prediction-table td.error-warn,
        .prediction-table td.error-bad {
            font-weight: 800;
        }

        .prediction-table td.error-good {
            background: #dcfce7 !important;
            color: #166534 !important;
        }

        .prediction-table td.error-warn {
            background: #fef3c7 !important;
            color: #92400e !important;
        }

        .prediction-table td.error-bad {
            background: #fee2e2 !important;
            color: #991b1b !important;
        }

        @media (prefers-color-scheme: dark) {
            .prediction-table td.error-good {
                background: #123524 !important;
                color: #86efac !important;
            }

            .prediction-table td.error-warn {
                background: #3a2f12 !important;
                color: #fde68a !important;
            }

            .prediction-table td.error-bad {
                background: #3b1717 !important;
                color: #fca5a5 !important;
            }
        }

        html[data-app-theme="dark"] .prediction-table td.error-good {
            background: #123524 !important;
            color: #86efac !important;
        }

        html[data-app-theme="dark"] .prediction-table td.error-warn {
            background: #3a2f12 !important;
            color: #fde68a !important;
        }

        html[data-app-theme="dark"] .prediction-table td.error-bad {
            background: #3b1717 !important;
            color: #fca5a5 !important;
        }

        html[data-app-theme="dark"] .js-plotly-plot .bg {
            fill: var(--panel) !important;
        }

        html[data-app-theme="dark"] .js-plotly-plot text {
            fill: var(--text) !important;
        }

        html[data-app-theme="dark"] .js-plotly-plot .gridlayer path,
        html[data-app-theme="dark"] .js-plotly-plot .zerolinelayer path {
            stroke: var(--line) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sync_streamlit_theme_marker() -> None:
    components.html(
        """
        <script>
        (function () {
            function isDarkColor(rgb) {
                const values = (rgb || "").match(/\\d+/g);
                if (!values || values.length < 3) return false;
                const r = Number(values[0]);
                const g = Number(values[1]);
                const b = Number(values[2]);
                return (r * 0.299 + g * 0.587 + b * 0.114) < 128;
            }

            function applyThemeMarker() {
                try {
                    const doc = window.parent.document;
                    const bodyBg = window.parent.getComputedStyle(doc.body).backgroundColor;
                    const theme = isDarkColor(bodyBg) ? "dark" : "light";
                    doc.documentElement.setAttribute("data-app-theme", theme);
                } catch (error) {
                    // No-op: the app still works with the default light variables.
                }
            }

            applyThemeMarker();
            window.setInterval(applyThemeMarker, 500);
        })();
        </script>
        """,
        height=0,
        width=0,
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


ERROR_COLOR_PALETTES = {
    "light": {
        "good": ("#dcfce7", "#166534"),
        "warn": ("#fef3c7", "#92400e"),
        "bad": ("#fee2e2", "#991b1b"),
    },
    "dark": {
        "good": ("#123524", "#86efac"),
        "warn": ("#3a2f12", "#fde68a"),
        "bad": ("#3b1717", "#fca5a5"),
    },
}


def current_theme_name() -> str:
    try:
        theme_type = getattr(st.context.theme, "type", None)
        if theme_type is None and hasattr(st.context.theme, "get"):
            theme_type = st.context.theme.get("type")
        if isinstance(theme_type, str) and theme_type.lower() == "dark":
            return "dark"
    except Exception:
        pass

    try:
        theme_base = st.get_option("theme.base")
        if isinstance(theme_base, str) and theme_base.lower() == "dark":
            return "dark"
    except Exception:
        pass

    return "light"


def error_pct_cell_style(value: float | int | None, theme_name: str = "light") -> str:
    if value is None or pd.isna(value):
        return ""

    palette = ERROR_COLOR_PALETTES.get(theme_name, ERROR_COLOR_PALETTES["light"])
    abs_error = abs(float(value))
    if abs_error < 2:
        bg_color, text_color = palette["good"]
    elif abs_error > 5:
        bg_color, text_color = palette["bad"]
    else:
        bg_color, text_color = palette["warn"]
    return f"background-color: {bg_color} !important; color: {text_color} !important; font-weight: 700;"


def prediction_error_class(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""

    abs_error = abs(float(value))
    if abs_error < 2:
        return "error-good"
    if abs_error > 5:
        return "error-bad"
    return "error-warn"


def prediction_table_html(predictions: pd.DataFrame) -> str:
    display_df = predictions.tail(160).copy()
    display_df = display_df.rename(
        columns={
            "date": "Ngày",
            "actual_close": "Giá thực tế",
            "predicted_close": "Giá dự báo",
            "error": "Sai số",
            "error_pct": "Sai số (%)",
        }
    )
    rows = []
    for index, row in display_df.iterrows():
        date_value = pd.Timestamp(row["Ngày"]).strftime("%Y-%m-%d")
        error_pct = row["Sai số (%)"]
        error_class = prediction_error_class(error_pct)
        rows.append(
            "<tr>"
            f"<td class='num'>{index}</td>"
            f"<td>{escape(date_value)}</td>"
            f"<td class='num'>{row['Giá thực tế']:,.2f}</td>"
            f"<td class='num'>{row['Giá dự báo']:,.2f}</td>"
            f"<td class='num'>{row['Sai số']:+,.2f}</td>"
            f"<td class='num {error_class}'>{error_pct:+.2f}%</td>"
            "</tr>"
        )

    return (
        "<div class='prediction-table-wrap'>"
        "<table class='prediction-table'>"
        "<thead><tr>"
        "<th></th>"
        "<th>Ngày</th>"
        "<th>Giá thực tế</th>"
        "<th>Giá dự báo</th>"
        "<th>Sai số</th>"
        "<th>Sai số (%)</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def chart_palette() -> dict[str, str]:
    if current_theme_name() == "dark":
        return {
            "panel": "#111827",
            "text": "#e5e7eb",
            "muted": "#94a3b8",
            "line": "#334155",
            "spike": "#94a3b8",
            "actual_line": "#f59e0b",
            "predicted_line": "#60a5fa",
        }
    return {
        "panel": "#ffffff",
        "text": "#17202a",
        "muted": "#667085",
        "line": "#d9e0ea",
        "spike": "#64748b",
        "actual_line": "#f97316",
        "predicted_line": "#2563eb",
    }


def header() -> None:
   st.markdown(
        """
        <div class="app-header">
            <div class="app-title">Dự báo giá cổ phiếu Việt Nam theo các nhóm ngành</div>
          
        </div>
        """,
        unsafe_allow_html=True,
    )
   


def warning_note() -> None:
    st.markdown(
        """
        <div class="notice">
            Kết quả dự báo chỉ phục vụ mục đích học tập nghiên cứu. Không phải khuyến nghị mua, bán hoặc nắm giữ chứng khoán.
        </div>
        """,
        unsafe_allow_html=True,
    )


CHART_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": False,
    "displaylogo": False,
    "doubleClick": "reset",
    "responsive": True,
}


def apply_chart_interactions(fig: go.Figure) -> go.Figure:
    palette = chart_palette()
    fig.update_layout(
        dragmode="pan",
        hovermode="x unified",
        clickmode="event+select",
        uirevision="keep",
        font=dict(color=palette["text"]),
        title_font=dict(color=palette["text"]),
        legend=dict(font=dict(color=palette["text"])),
    )
    fig.update_xaxes(
        color=palette["text"],
        title_font=dict(color=palette["text"]),
        tickfont=dict(color=palette["muted"]),
        gridcolor=palette["line"],
        zerolinecolor=palette["line"],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor=palette["spike"],
    )
    fig.update_yaxes(
        color=palette["text"],
        title_font=dict(color=palette["text"]),
        tickfont=dict(color=palette["muted"]),
        gridcolor=palette["line"],
        zerolinecolor=palette["line"],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor=palette["spike"],
    )
    return fig


def chart_title(title: str) -> dict[str, object]:
    palette = chart_palette()
    return {
        "text": title,
        "x": 0.01,
        "y": 0.98,
        "xanchor": "left",
        "yanchor": "top",
        "font": {"size": 20, "color": palette["text"]},
    }


def sync_data_panel(raw_df: pd.DataFrame, current_symbol: str) -> None:
    with st.sidebar.expander("Đồng bộ dữ liệu"):
        st.caption("Dữ liệu OHLCV được cập nhật mặc định từ vnstock.vn")
        scope = st.radio(
            "Phạm vi cập nhật",
            ["Mã hiện tại", "Tất cả các mã hiện tại"],
            label_visibility="visible",
        )

        today = date.today()
        default_start = date(2015, 1, 1)

        start_date = st.date_input("Từ ngày", value=default_start)
        end_date = st.date_input("Đến ngày", value=today)
        st.caption("Để lấy đủ dữ liệu lịch sử cho mô hình, nên mốc trước ngày 2020-01-01.")

        if scope == "Mã đang chọn":
            symbols_to_sync = [current_symbol]
        else:
            symbols_to_sync = [symbol for symbols in INDUSTRY_GROUPS.values() for symbol in symbols]

        st.caption("Mã cập nhật: " + ", ".join(symbols_to_sync))

        if st.button("Đồng bộ dữ liệu ", use_container_width=True):
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
    palette = chart_palette()
    fig = go.Figure()
    customdata = df[["open", "high", "low", "volume"]].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            customdata=customdata,
            mode="lines+markers",
            name="Giá đóng cửa",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=4, color="#2563eb", opacity=0.7),
            hovertemplate=(
                "Ngày: %{x|%Y-%m-%d}<br>"
                "O: %{customdata[0]:,.2f}<br>"
                "H: %{customdata[1]:,.2f}<br>"
                "L: %{customdata[2]:,.2f}<br>"
                "C: %{y:,.2f}<br>"
                "V: %{customdata[3]:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=chart_title(title),
        height=430,
        margin=dict(l=16, r=16, t=72, b=16),
       # xaxis_title="Ngày",
      #  yaxis_title="Giá đóng cửa",
        paper_bgcolor=palette["panel"],
        plot_bgcolor=palette["panel"],
    )
    return apply_chart_interactions(fig)


def candlestick_chart(df: pd.DataFrame, title: str) -> go.Figure:
    palette = chart_palette()
    customdata = df[["volume"]].to_numpy()
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                customdata=customdata,
                name="OHLC",
                increasing_line_color="#16803c",
                decreasing_line_color="#c62828",
                hovertemplate=(
                    "Ngày: %{x|%Y-%m-%d}<br>"
                    "O: %{open:,.2f}<br>"
                    "H: %{high:,.2f}<br>"
                    "L: %{low:,.2f}<br>"
                    "C: %{close:,.2f}<br>"
                    "V: %{customdata[0]:,.0f}"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=chart_title(title),
        height=430,
        margin=dict(l=16, r=16, t=72, b=16),
     #   xaxis_title="Ngày",
      #  yaxis_title="Giá",
        paper_bgcolor=palette["panel"],
        plot_bgcolor=palette["panel"],
        xaxis_rangeslider_visible=False,
    )
    return apply_chart_interactions(fig)


def sector_comparison_chart(df: pd.DataFrame, title: str) -> go.Figure:
    palette = chart_palette()
    normalized = normalize_sector_prices(df)
    fig = go.Figure()
    for symbol, group in normalized.groupby("symbol"):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["normalized_close"],
                customdata=group[["close"]].to_numpy(),
                mode="lines",
                name=symbol,
                hovertemplate=(
                    "Ngày: %{x|%Y-%m-%d}<br>"
                    "Giá đóng cửa: %{customdata[0]:,.2f}<br>"
                    "Giá chuẩn hóa: %{y:,.2f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_layout(
        title=chart_title(title),
        height=430,
        margin=dict(l=16, r=16, t=72, b=16),
     #   xaxis_title="Ngày",
     #   yaxis_title="Giá chuẩn hóa",
        paper_bgcolor=palette["panel"],
        plot_bgcolor=palette["panel"],
    )
    return apply_chart_interactions(fig)


def prediction_chart(predictions: pd.DataFrame, title: str) -> go.Figure:
    palette = chart_palette()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["actual_close"],
            mode="lines+markers",
            name="Giá thực tế",
            line=dict(color=palette["actual_line"], width=2),
            marker=dict(size=4, color=palette["actual_line"], opacity=0.8),
            hovertemplate=(
                "Ngày: %{x|%Y-%m-%d}<br>"
                "Giá thực tế: %{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["predicted_close"],
            customdata=predictions[["error", "error_pct"]].to_numpy(),
            mode="lines+markers",
            name="Giá dự báo",
            line=dict(color=palette["predicted_line"], width=2),
            marker=dict(size=4, color=palette["predicted_line"], opacity=0.8),
            hovertemplate=(
                "Ngày: %{x|%Y-%m-%d}<br>"
                "Giá dự báo: %{y:,.2f}<br>"
                "Sai số: %{customdata[0]:,.2f}<br>"
                "Sai số %: %{customdata[1]:,.2f}%"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=chart_title(title),
        height=430,
        margin=dict(l=16, r=16, t=72, b=16),
 #       xaxis_title="Ngày",
  #      yaxis_title="Giá đóng cửa",
        paper_bgcolor=palette["panel"],
        plot_bgcolor=palette["panel"],
    )
    return apply_chart_interactions(fig)


def show_overview(sector_df: pd.DataFrame, symbol_df: pd.DataFrame, symbol: str) -> None:
    latest = symbol_df.iloc[-1]
    first = symbol_df.iloc[0]
    latest_date = pd.Timestamp(latest["date"]).date().isoformat()
    total_return = (latest["close"] / first["close"] - 1) * 100 if first["close"] else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mã cổ phiếu", symbol)
    col2.metric(
        "Giá mới nhất",
        fmt_price(latest["close"]),
        help=f"Giá đóng cửa tính đến ngày cập nhật dữ liệu gần đây nhất: {latest_date}.",
    )
    col3.metric("Số phiên dữ liệu", fmt_int(len(symbol_df)))
    col4.metric("Tăng/giảm toàn kỳ", fmt_pct(total_return))
   

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.plotly_chart(
            price_line_chart(symbol_df, f"<b>Diễn biến giá đóng cửa - {symbol}</b>"),
            use_container_width=True,
            config=CHART_CONFIG,
            theme=None,
        )
    with c2:
        st.plotly_chart(
            sector_comparison_chart(sector_df, "<b>So sánh cổ phiếu cùng nhóm ngành</b>"),
            use_container_width=True,
            config=CHART_CONFIG,
            theme=None,
        )

    st.plotly_chart(
        candlestick_chart(symbol_df, f"<b>Biểu đồ nến - {symbol}</b>"),
        use_container_width=True,
        config=CHART_CONFIG,
        theme=None,
    )

    st.subheader("Bảng kiểm tra dữ liệu")
    st.dataframe(validate_dataset(sector_df), use_container_width=True, hide_index=True)

    st.subheader(f"Dữ liệu gần nhất của {symbol}(120 phiên)" )
    st.dataframe(symbol_df.tail(120), use_container_width=True, hide_index=True)


def show_features(feature_df: pd.DataFrame, symbol_features: pd.DataFrame, symbol: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"""
        <div class="feature-summary-card">
            <div class="feature-summary-label">Số đặc trưng đầu vào</div>
            <div class="feature-summary-value">{fmt_int(len(FEATURE_COLUMNS))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class="feature-summary-card">
            <div class="feature-summary-label">Số dòng đặc trưng của mã</div>
            <div class="feature-summary-value">{fmt_int(len(symbol_features))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        """
        <div class="feature-summary-card">
            <div class="feature-summary-label">Mục tiêu dự báo</div>
            <div class="feature-summary-value text-value">Giá đóng cửa phiên kế tiếp</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col4.markdown(
        f"""
        <div class="feature-summary-card">
            <div class="feature-summary-label">Số mẫu train hợp lệ</div>
            <div class="feature-summary-value">{fmt_int(symbol_features[TARGET_COLUMN].notna().sum())}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Mục tiêu chính của mô hình là dự báo giá đóng cửa của phiên giao dịch kế tiếp. "
        "Biến tăng/giảm chỉ dùng để diễn giải xu hướng, không phải giá trị dự báo chính của mô hình."
    )
    
    st.subheader(f"Bảng các giá trị đặc trưng gần nhất - {symbol}")
    expected_columns = ["date", "symbol", *FEATURE_COLUMNS, TARGET_COLUMN, "target_return", "target_up"]
    display_columns = [column for column in expected_columns if column in symbol_features.columns]
    missing_display_columns = [column for column in expected_columns if column not in symbol_features.columns]

    if missing_display_columns:
        st.warning(
            "Một số cột chưa có trong dữ liệu đặc trưng nên được ẩn khỏi bảng hiển thị: "
            + ", ".join(missing_display_columns)
        )

    display_df = symbol_features[display_columns].tail(120).rename(
        columns={
            "date": "Ngày",
            "symbol": "Mã",
            TARGET_COLUMN: "Giá đóng cửa phiên kế tiếp",
            "target_return": "Tỷ suất biến động phiên kế tiếp",
            "target_up": "Xu hướng phiên kế tiếp (1=tăng, 0=giảm)",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)




    st.subheader("Danh sách các đặc trưng")
    st.dataframe(feature_description_table(), use_container_width=True, hide_index=True)

  



def show_modeling(symbol_features: pd.DataFrame, model_name: str, test_size: float, symbol: str) -> None:
    train_count = symbol_features[TARGET_COLUMN].notna().sum()
    st.write(f"Dữ liệu huấn luyện khả dụng cho {symbol}: {train_count} phiên.")
  
    if model_name == "LSTM":
        st.caption(
            "LSTM dùng chuỗi 20 phiên gần nhất để học quan hệ theo thời gian. "
            "Do đó, số mẫu huấn luyện thực tế sẽ giảm đi so với tổng số dòng có giá trị mục tiêu."
        )

    if st.button("Huấn luyện mô hình và dự báo", use_container_width=True):
        try:
            with st.spinner("Đang huấn luyện mô hình..."):
                st.session_state["training_result"] = train_and_predict(symbol_features, model_name, test_size)
                st.session_state["training_key"] = (symbol, model_name, test_size, TRAINING_PIPELINE_VERSION)
        except Exception as exc:
            st.error(f"Không thể huấn luyện mô hình: {exc}")
            return

    result = st.session_state.get("training_result")
    result_key = st.session_state.get("training_key")
    current_key = (symbol, model_name, test_size, TRAINING_PIPELINE_VERSION)

    if result is None or result_key != current_key:
        st.info("Bấm nút huấn luyện để xem kết quả đánh giá và dự báo.")
        return

    metrics = result.metrics
    next_prediction = result.next_prediction
    predictions = result.predictions

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("MSE", f"{metrics['MSE']:,.4f}")
    col2.metric("MAE", fmt_price(metrics["MAE"]))
    col3.metric("RMSE", fmt_price(metrics["RMSE"]))
    col4.metric("MAPE", fmt_pct(metrics["MAPE"]))
    col5.metric("SMAPE", fmt_pct(metrics["SMAPE"]))
    col6.metric("R2", f"{metrics['R2']:.4f}")

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
        prediction_chart(predictions, f"So sánh giá đóng cửa thực tế và dự báo của {symbol}"),
        use_container_width=True,
        config=CHART_CONFIG,
        theme=None,
    )

    st.subheader("Bảng kết quả trên tập kiểm thử")
    st.caption("Chú thích: Màu ở cột Sai số (%): xanh < 2%, vàng từ 2% đến 5%, đỏ > 5%.")
    st.markdown(prediction_table_html(predictions), unsafe_allow_html=True)

    csv_data = predictions.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Tải kết quả dự báo dưới dạng file CSV",
        data=csv_data,
        file_name=f"du_bao_{symbol}_{model_name.replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )



def main() -> None:
    inject_css()
    sync_streamlit_theme_marker()
    header()
    warning_note()

    try:
        raw_df = load_data()
        feature_df = build_features(raw_df)
    except Exception as exc:
        st.error(f"Không thể khởi tạo ứng dụng: {exc}")
        st.stop()

    with st.sidebar:
        st.markdown('<div class="sidebar-menu-title">Menu</div>', unsafe_allow_html=True)
        industry = st.selectbox("Nhóm ngành", list(INDUSTRY_GROUPS.keys()))
        available_symbols = INDUSTRY_GROUPS[industry]
        symbol = st.selectbox("Mã cổ phiếu", available_symbols)
        model_name = st.selectbox("Mô hình dự báo", MODEL_OPTIONS)
        test_size = st.slider("Tỷ lệ dữ liệu test", 0.10, 0.40, 0.20, 0.05)

        sync_data_panel(raw_df, symbol)

    sector_df = filter_industry(raw_df, industry)
    symbol_df = filter_symbol(raw_df, symbol)
    symbol_features = filter_symbol(feature_df, symbol)

    if symbol_df.empty or symbol_features.empty:
        st.error("Không đủ dữ liệu cho mã cổ phiếu đã chọn.")
        st.stop()

    tab_overview, tab_features, tab_model = st.tabs(
        ["Tổng quan", "Đặc trưng", "Huấn luyện và dự báo"]
    )

    with tab_overview:
        show_overview(sector_df, symbol_df, symbol)

    with tab_features:
        show_features(feature_df, symbol_features, symbol)

    with tab_model:
        show_modeling(symbol_features, model_name, test_size, symbol)

   


if __name__ == "__main__":
    main()
