# Dự báo giá cổ phiếu Việt Nam theo nhóm ngành

Đây là ứng dụng Streamlit phục vụ đồ án 1 theo hướng ứng dụng. Ứng dụng dùng dữ liệu lịch sử cổ phiếu Việt Nam trong `data/data.csv`, tạo đặc trưng kỹ thuật, huấn luyện mô hình học máy và dự báo giá đóng cửa của phiên giao dịch kế tiếp.

## Dữ liệu

File chính:

```text
data/data.csv
```

Các cột bắt buộc:

```text
date, open, high, low, close, volume, symbol
```

Các mã đang có trong dữ liệu:

- Ngân hàng: `ACB`, `BID`, `VCB`
- Công nghệ: `FPT`
- Tiêu dùng: `VNM`

Ứng dụng không tải dữ liệu từ internet khi chạy, nên phù hợp để demo ổn định trước giảng viên.

## Chức năng

- Chọn nhóm ngành và mã cổ phiếu.
- Đồng bộ dữ liệu từ `vnstock` cho mã đang chọn hoặc toàn bộ mã trong ứng dụng.
- Xem tổng quan dữ liệu, bảng dữ liệu, biểu đồ giá đóng cửa và biểu đồ nến.
- So sánh giá chuẩn hóa của các cổ phiếu trong cùng nhóm ngành.
- Tạo đặc trưng kỹ thuật:
  - Giá và khối lượng: `open`, `high`, `low`, `close`, `volume`
  - Biến động trong phiên: `price_change`, `price_change_pct`, `high_low_pct`
  - Xu hướng: `return`, `ma5`, `ma10`, `ma20`, `ma50`
  - Tỷ lệ giá so với MA: `close_ma5_ratio`, `close_ma20_ratio`, `close_ma50_ratio`
  - Biến trễ: `close_lag1`, `close_lag2`, `close_lag3`, `volume_lag1`
  - Biến động: `volume_ma20`, `volatility_10`
- Huấn luyện các mô hình:
  - Linear Regression
  - Random Forest
  - SVR
- Đánh giá bằng MAE, RMSE, MAPE và R2.
- Dự báo giá đóng cửa phiên giao dịch kế tiếp.
- Tải kết quả dự báo trên tập kiểm thử ra file CSV.

## Cài đặt và chạy

Tạo môi trường ảo nếu cần:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy ứng dụng:

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại:

```text
http://localhost:8501
```

## Cấu trúc code

```text
app.py          Giao diện Streamlit và luồng thao tác chính
features.py     Đọc dữ liệu, kiểm tra dữ liệu, tạo đặc trưng
modeling.py     Huấn luyện mô hình, đánh giá, dự báo phiên kế tiếp
data/data.csv   Dữ liệu cổ phiếu dùng cho demo
data_sync.py    Đồng bộ dữ liệu mới từ vnstock và ghi lại data/data.csv
```

## Đồng bộ dữ liệu từ vnstock

Trong sidebar, mở mục **Đồng bộ dữ liệu vnstock**, chọn khoảng ngày và bấm **Đồng bộ dữ liệu từ vnstock**.

Ứng dụng sẽ:

- Lấy dữ liệu OHLCV từ vnstock.
- Chuẩn hóa cột về `date, open, high, low, close, volume, symbol`.
- Gộp với dữ liệu cũ trong `data/data.csv`.
- Loại trùng theo `symbol + date`.
- Tải lại dữ liệu trong app sau khi đồng bộ.

Nếu máy chưa có thư viện `vnstock`, chạy lại:

```bash
pip install -r requirements.txt
```

## Lưu ý khi trình bày

Kết quả dự báo chỉ phục vụ mục đích học tập, không phải khuyến nghị đầu tư. Khi bảo vệ, nên trình bày rõ quy trình: dữ liệu đầu vào, đặc trưng đã chọn, mô hình sử dụng, cách chia train/test, chỉ số đánh giá và kết quả dự báo.
