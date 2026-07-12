"""Ingest monthly commodity prices from the World Bank Pink Sheet.

Source page (get the current .xlsx link from here — the World Bank rotates the
document id every year, so do NOT hard-code a stale URL):
    https://www.worldbank.org/en/research/commodity-markets
File: "Monthly prices" -> CMO-Historical-Data-Monthly.xlsx

Usage:
    python -m ingest.pink_sheet --url  https://.../CMO-Historical-Data-Monthly.xlsx
    python -m ingest.pink_sheet --file data/CMO-Historical-Data-Monthly.xlsx

The parser locates the header row by searching for the commodity labels rather
than assuming a fixed row index, so a layout change upstream does not break it
silently — it raises instead.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from datetime import date

import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

# Windows console mặc định dùng cp1252 -> vỡ khi in tiếng Việt. Ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.db import get_engine, init_db, log_ingest, upsert_prices, upsert_series  # noqa: E402

SHEET_HINT = "monthly"   # tự dò sheet có chữ "monthly" trong tên
SOURCE_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
LICENSE = "World Bank Open Data — dùng lại kèm ghi nguồn (CC BY 4.0)"

# series_id -> (nhãn tiếng Việt, các cách viết tên cột có thể gặp trong Pink Sheet)
# So khớp sau khi chuẩn hoá: bỏ dấu cách thừa, không phân biệt hoa/thường.
SERIES = {
    "coffee_robusta": ("Cà phê Robusta (chỉ báo ICO)",
                       ["Coffee, Robusta"]),
    "coffee_arabica": ("Cà phê Arabica (chỉ báo ICO, other mild Arabicas)",
                       ["Coffee, Arabica"]),
    "cocoa":          ("Ca cao",
                       ["Cocoa"]),
    "rubber_tsr20":   ("Cao su",
                       ["Rubber, TSR20 **"]),
}


def _norm(x) -> str:
    return " ".join(str(x).strip().lower().split())

DATE_RE = re.compile(r"^\s*(\d{4})M(\d{1,2})\s*$")


def _pick_sheet(xls: pd.ExcelFile) -> str:
    """Dò sheet giá theo tháng. Không khoá cứng tên -> không vỡ khi WB đổi nhãn."""
    for name in xls.sheet_names:
        if SHEET_HINT in name.lower():
            return name
    raise RuntimeError(
        f"Không thấy sheet nào chứa '{SHEET_HINT}'. Các sheet có trong file: "
        f"{xls.sheet_names}.\nCó thể bạn đang tải nhầm bản tin Pink Sheet hằng "
        f"tháng (PDF/xlsx chỉ vài tháng gần nhất) thay vì file lịch sử "
        f"CMO-Historical-Data-Monthly.xlsx."
    )


def _load_frame(url: str | None, path: str | None) -> tuple[pd.DataFrame, str]:
    src = path
    if not path:
        resp = requests.get(url, timeout=120, headers={"User-Agent": "ADE-Agri/1.0"})
        resp.raise_for_status()
        src = io.BytesIO(resp.content)
    xls = pd.ExcelFile(src)
    sheet = _pick_sheet(xls)
    print(f"  sheet: '{sheet}'")
    return pd.read_excel(xls, sheet_name=sheet, header=None), sheet


def _find_header_row(df: pd.DataFrame, sheet_name: str = "") -> int:
    """Dòng chứa tên mặt hàng. Luôn dò, không bao giờ giả định vị trí."""
    for i in range(min(25, len(df))):
        cells = {_norm(c) for c in df.iloc[i].tolist()}
        if any(_norm(a) in cells for _, aliases in SERIES.values() for a in aliases):
            return i
    raise RuntimeError(
        f"Không tìm thấy dòng tiêu đề chứa tên mặt hàng trong sheet '{sheet_name}'. "
        "Pink Sheet có thể đã đổi cấu trúc — kiểm tra lại file."
    )


def parse(df: pd.DataFrame, sheet_name: str = ""):
    hdr = _find_header_row(df, sheet_name)
    names = [_norm(c) for c in df.iloc[hdr].tolist()]

    raw_names = [str(c).strip() for c in df.iloc[hdr].tolist()]
    col_of: dict[str, int] = {}
    matched: dict[str, str] = {}      # sid -> tên cột THẬT trong file
    missing: list[str] = []
    for sid, (_, aliases) in SERIES.items():
        hit = next((_norm(a) for a in aliases if _norm(a) in names), None)
        if hit is None:
            missing.append(sid)
            continue
        i = names.index(hit)
        col_of[sid] = i
        matched[sid] = raw_names[i]

    if missing:
        real = [n for n in names if n and n != "nan"]
        print("\n[CẢNH BÁO] Không tìm thấy cột cho: " + ", ".join(missing))
        print("Các cột CÓ THẬT trong file (dùng để đối chiếu):")
        for n in real:
            print("   -", n)
        print("Các chuỗi còn lại vẫn được nạp bình thường.\n")

    if not col_of:
        raise RuntimeError("Không khớp được cột nào. Xem danh sách cột ở trên.")

    units_row = [str(c).strip() for c in df.iloc[hdr + 1].tolist()]
    units = {sid: units_row[i] for sid, i in col_of.items()}

    out: dict[str, list[tuple[date, float]]] = {sid: [] for sid in col_of}
    for _, row in df.iloc[hdr + 1:].iterrows():
        m = DATE_RE.match(str(row.iloc[0]))
        if not m:
            continue
        obs = date(int(m.group(1)), int(m.group(2)), 1)
        for sid, ci in col_of.items():
            val = pd.to_numeric(row.iloc[ci], errors="coerce")
            if pd.notna(val):
                out[sid].append((obs, float(val)))

    return out, units, matched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="URL của CMO-Historical-Data-Monthly.xlsx")
    ap.add_argument("--file", help="Đường dẫn file .xlsx đã tải sẵn")
    ap.add_argument("--db", help="DATABASE_URL (mặc định: sqlite cục bộ)")
    args = ap.parse_args()

    if not args.url and not args.file:
        ap.error("cần --url hoặc --file")

    engine = get_engine(args.db)
    init_db(engine)

    try:
        df, sheet = _load_frame(args.url, args.file)
        data, units, matched = parse(df, sheet)
    except Exception as exc:  # noqa: BLE001
        log_ingest(engine, "pink_sheet", "error", message=str(exc))
        print(f"[LỖI] {exc}", file=sys.stderr)
        return 1

    seen = added = updated = 0
    for sid, rows in data.items():
        # nhãn luôn kèm tên cột gốc -> không bao giờ dán nhãn sai loại hàng
        label_vi = f"{SERIES[sid][0]} — {matched[sid]}"
        upsert_series(engine, {
            "series_id": sid,
            "label_vi": label_vi,
            "unit": units.get(sid, ""),
            "frequency": "monthly",
            "source_name": "World Bank — Commodity Price Data (Pink Sheet)",
            "source_url": SOURCE_PAGE,
            "license_note": LICENSE,
            "is_active": 1 if rows else 0,
        })
        a, u = upsert_prices(engine, sid, rows)
        seen += len(rows)
        added += a
        updated += u
        span = f"{rows[0][0]} → {rows[-1][0]}" if rows else "trống"
        print(f"  {sid:16s} {len(rows):5d} quan sát  [{span}]  +{a} mới, ~{u} cập nhật")

    log_ingest(engine, "pink_sheet", "ok", seen=seen, added=added, updated=updated,
               message=f"{len(data)}/{len(SERIES)} chuỗi")
    print(f"\nXong: {seen} quan sát, {added} dòng mới, {updated} dòng cập nhật.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())