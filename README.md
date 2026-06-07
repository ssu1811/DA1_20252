# Dự báo giá cổ phiếu Việt Nam theo nhóm ngành

Đây là ứng dụng Streamlit phục vụ đồ án theo hướng ứng dụng. Ứng dụng dùng dữ liệu lịch sử cổ phiếu Việt Nam trong `data/data.csv`, tạo đặc trưng kỹ thuật, huấn luyện mô hình dự báo và dự báo giá đóng cửa của phiên giao dịch kế tiếp.

## Dữ liệu

File dữ liệu chính:

```text
data/data.csv
```

Các cột bắt buộc:

```text
date, open, high, low, close, volume, symbol
```

Các nhóm ngành trong ứng dụng:

- Ngân hàng: `ACB`, `BID`, `VCB`
- Công nghệ: `FPT`, `CMG`, `ELC`
- Tiêu dùng: `VNM`, `SAB`, `MSN`

Ứng dụng đọc dữ liệu từ CSV để demo ổn định. Nếu cần cập nhật dữ liệu mới, dùng mục **Đồng bộ dữ liệu** trong sidebar hoặc chạy `data_sync.py`.

## Chức năng chính

- Chọn nhóm ngành, mã cổ phiếu, mô hình dự báo và tỷ lệ test.
- Đồng bộ dữ liệu OHLCV từ `vnstock` cho mã đang chọn hoặc toàn bộ mã trong ứng dụng.
- Xem tổng quan dữ liệu, bảng kiểm tra dữ liệu, biểu đồ giá đóng cửa và biểu đồ nến.
- So sánh giá chuẩn hóa của các cổ phiếu trong cùng nhóm ngành.
- Tạo đặc trưng kỹ thuật từ giá, khối lượng, xu hướng, biến trễ và biến động.
- Huấn luyện các mô hình:
  - Linear Regression
  - Random Forest
  - SVR
  - LSTM
- Đánh giá bằng MSE, MAE, RMSE, MAPE, SMAPE và R2.
- Dự báo giá đóng cửa phiên giao dịch kế tiếp.
- Tải kết quả dự báo trên tập kiểm thử ra file CSV.

## Đặc trưng sử dụng

- Giá và khối lượng: `open`, `high`, `low`, `close`, `volume`
- Biến động trong phiên: `price_change`, `price_change_pct`, `high_low_pct`
- Xu hướng: `return`, `ma5`, `ma10`, `ma20`, `ma50`
- Tỷ lệ giá so với đường trung bình: `close_ma5_ratio`, `close_ma20_ratio`, `close_ma50_ratio`
- Biến trễ: `close_lag1`, `close_lag2`, `close_lag3`, `volume_lag1`
- Biến động: `volume_ma20`, `volatility_10`
- Biến mục tiêu:
  - `target_next_close`: giá đóng cửa phiên kế tiếp
  - `target_return`: tỷ suất thay đổi của phiên kế tiếp
  - `target_up`: phiên kế tiếp tăng hay giảm

Random Forest, SVR và LSTM học `target_return`, sau đó quy đổi lại thành giá đóng cửa dự báo. Cách này giúp mô hình bám tốt hơn vào biến động tương đối của cổ phiếu.

## Cài đặt và chạy

Tạo môi trường ảo nếu cần:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Cài thư viện cơ bản:

```bash
pip install -r requirements.txt
```

Nếu muốn chạy mô hình LSTM, cài thêm TensorFlow:

```bash
pip install -r requirements-lstm.txt
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
app.py                  Giao diện Streamlit và luồng thao tác chính
features.py             Đọc dữ liệu, kiểm tra dữ liệu, tạo đặc trưng
modeling.py             Huấn luyện mô hình, đánh giá, dự báo phiên kế tiếp
data_sync.py            Đồng bộ dữ liệu mới từ vnstock
data/data.csv           Dữ liệu cổ phiếu dùng cho demo
requirements.txt        Thư viện cơ bản
requirements-lstm.txt   Thư viện bổ sung cho mô hình LSTM
```

## Lưu ý khi trình bày

Kết quả dự báo chỉ phục vụ mục đích học tập, không phải khuyến nghị đầu tư. Khi bảo vệ, nên trình bày rõ quy trình: dữ liệu đầu vào, đặc trưng đã chọn, biến mục tiêu, cách chia train/test, mô hình sử dụng, chỉ số đánh giá và kết quả dự báo.
