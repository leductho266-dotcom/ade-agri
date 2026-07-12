# ADE-Agri — lớp dữ liệu

Thu thập, chuẩn hoá và lưu trữ chuỗi giá nông sản chủ lực phục vụ hệ thống
cảnh báo sớm biến động giá tỉnh Đắk Lắk.

## Chạy lần đầu (máy cá nhân)

```bash
pip install -r requirements.txt

# 1. Vào https://www.worldbank.org/en/research/commodity-markets
#    tải "Monthly prices" -> CMO-Historical-Data-Monthly.xlsx
# 2. Chạy:
python -m ingest.pink_sheet --file CMO-Historical-Data-Monthly.xlsx
```

Không cần cấu hình gì thêm: mặc định tạo file SQLite `ade_agri.db`.

Chạy lại nhiều lần **không** sinh dòng trùng (khoá chính `series_id + obs_date`).

## Chuyển sang Supabase (khi deploy)

```bash
export DATABASE_URL="postgresql+psycopg://postgres:...@...:5432/postgres"
python -m ingest.pink_sheet --url "<link .xlsx>"
```

## Tự động cập nhật

`.github/workflows/ingest.yml` chạy mỗi ngày. Cần khai báo 2 secret trong
repo (Settings > Secrets and variables > Actions):

- `DATABASE_URL`
- `PINK_SHEET_URL`

Nhật ký chạy của workflow là bằng chứng công khai, có dấu thời gian, rằng dữ
liệu được cập nhật thật — dùng cho trang "Nguồn dữ liệu" của ứng dụng.

## Chuỗi hiện có

| series_id | Mặt hàng | Nguồn | Tần suất |
|---|---|---|---|
| `coffee_robusta` | Cà phê Robusta (chỉ báo ICO) | World Bank Pink Sheet | tháng |
| `cocoa` | Ca cao | World Bank Pink Sheet | tháng |
| `rubber_tsr20` | Cao su TSR20 | World Bank Pink Sheet | tháng |
| _hồ tiêu_ | **chưa có** — đang xác lập nguồn (IPC / VPSA) | | |
| _giá nội địa Đắk Lắk_ | **chưa có** — cần thu thập hằng ngày | | |

Ghi nguồn bắt buộc khi hiển thị: World Bank, Commodity Price Data (The Pink
Sheet), CC BY 4.0.

## Bảng dữ liệu

`series` · `prices` · `forecasts` · `weights` · `metrics` · `ingest_log`

Xem `schema.sql`. Hai điểm thiết kế không thương lượng:

- `forecasts` luôn có `lo`/`hi` — không bao giờ lưu điểm dự báo trần trụi.
- `metrics` có cột `model` và `regime` — chỗ để đặt baseline (bước ngẫu nhiên,
  trung bình tổ hợp, từng mô hình đơn) cạnh ADE, tách theo giai đoạn thị trường.
