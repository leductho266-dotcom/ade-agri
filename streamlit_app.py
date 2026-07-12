"""ADE-Agri — khung ứng dụng (bước deploy đầu tiên).

Mục đích của file này KHÔNG phải là dashboard. Mục đích là để lộ toàn bộ rủi ro
hạ tầng NGAY BÂY GIỜ: kết nối Supabase, secrets, phiên bản package, cold start.
Khi trang này hiện màu xanh trên URL công khai, mọi việc còn lại chỉ là nội dung.

Nguyên tắc: ứng dụng CHỈ ĐỌC. Không train, không import tensorflow/pmdarima.
Mô hình chạy ở GitHub Actions và ghi vào Supabase; app chỉ SELECT.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="ADE-Agri", page_icon="📈", layout="wide")


def _database_url() -> str | None:
    """Streamlit Cloud -> st.secrets; local -> biến môi trường."""
    try:
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    return os.environ.get("DATABASE_URL")


@st.cache_resource
def get_engine():
    url = _database_url()
    if not url:
        return None
    return create_engine(url, future=True, pool_pre_ping=True)


@st.cache_data(ttl=600)
def query(sql: str) -> pd.DataFrame:
    eng = get_engine()
    if eng is None:
        return pd.DataFrame()
    with eng.connect() as conn:
        return pd.DataFrame(conn.execute(text(sql)).mappings().all())


st.title("ADE-Agri")
st.caption("Hệ thống cảnh báo sớm biến động giá nông sản chủ lực — tỉnh Đắk Lắk")

st.divider()
st.subheader("Kiểm tra kết nối")

if get_engine() is None:
    st.error(
        "Chưa cấu hình DATABASE_URL.\n\n"
        "• Streamlit Cloud: Settings → Secrets → thêm dòng\n"
        "  `DATABASE_URL = \"postgresql+psycopg://...\"`\n"
        "• Local: đặt biến môi trường DATABASE_URL"
    )
    st.stop()

try:
    counts = {}
    for tbl in ("series", "prices", "forecasts", "weights", "metrics",
                "dm_tests", "ingest_log"):
        df = query(f"SELECT COUNT(*) AS n FROM {tbl}")
        counts[tbl] = int(df.iloc[0]["n"]) if not df.empty else 0

    st.success("Kết nối cơ sở dữ liệu thành công.")

    cols = st.columns(len(counts))
    for col, (tbl, n) in zip(cols, counts.items()):
        col.metric(tbl, f"{n:,}")

    empty = [t for t, n in counts.items() if n == 0]
    if empty:
        st.warning(
            "Bảng chưa có dữ liệu: " + ", ".join(empty) +
            " — chạy `ingest.pink_sheet` rồi `model.ade_runner` trỏ vào Supabase."
        )

    st.divider()
    st.subheader("Các chuỗi đang theo dõi")
    s = query(
        "SELECT series_id, label_vi, unit, frequency, source_name, is_active "
        "FROM series ORDER BY series_id"
    )
    if s.empty:
        st.info("Bảng `series` trống.")
    else:
        st.dataframe(s, use_container_width=True, hide_index=True)

    st.subheader("Dự báo mới nhất (mẫu — chưa phải giao diện thật)")
    f = query(
        "SELECT series_id, run_date, target_date, horizon, point, lo, hi "
        "FROM forecasts WHERE model = 'ADE' "
        "ORDER BY run_date DESC, series_id, horizon LIMIT 20"
    )
    if f.empty:
        st.info("Bảng `forecasts` trống.")
    else:
        st.dataframe(f, use_container_width=True, hide_index=True)

except Exception as exc:  # noqa: BLE001
    st.error(f"Kết nối được nhưng truy vấn lỗi:\n\n```\n{exc}\n```")
    st.info("Thường là do schema chưa được tạo trên Supabase. "
            "Chạy `init_db()` hoặc dán `schema.sql` vào SQL Editor của Supabase.")

st.divider()
st.caption(
    "Nguồn dữ liệu: World Bank — Commodity Price Data (Pink Sheet), CC BY 4.0. "
    "Đây là bản dựng kỹ thuật, chưa phải sản phẩm hoàn chỉnh. "
    "Dự báo mang tính tham khảo, không phải khuyến nghị mua bán."
)
