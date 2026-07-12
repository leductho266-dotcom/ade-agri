"""ADE-Agri — chạy ADE-NoSTL trên chuỗi giá hàng hoá, ghi kết quả vào CSDL.

    python -m model.ade_runner                                    # 3 chuỗi, h = 1,3,6,12
    python -m model.ade_runner --series coffee_robusta --horizons 1

Thiết kế — bốn điểm khác pipeline luận án, mỗi điểm có lý do:

1. BASELINE RANDOM WALK. Với giá hàng hoá, bước ngẫu nhiên là đối thủ mạnh nhất.
   Không đối chứng với nó thì không có bằng chứng nào cả.

2. KHOẢNG DỰ BÁO. Phân vị thực nghiệm của sai số ADE trên tập VALIDATION, theo
   từng horizon. Không giả định phân phối chuẩn. Không đụng vào tập test.

3. HAI QUY TẮC PHÂN CHẾ ĐỘ, ĐỊNH NGHĨA TRƯỚC, BÁO CÁO CẢ HAI.
     A · jump  — |Δlog y_t| > BREAK_K · σ_train        (cú sốc một tháng)
     B · vol   — σ trượt 12 tháng > phân vị VOL_Q trên train  (biến động kéo dài)
   Hai quy tắc đo hai hiện tượng khác nhau. Ngưỡng ước lượng CHỈ trên train.
   Cấm chọn quy tắc cho kết quả đẹp hơn rồi giấu quy tắc kia — đó là p-hacking.

4. DỰ BÁO TIẾN. Backtest dừng ở quan sát cuối; dashboard cần dự báo cho tháng
   chưa xảy ra.

Chia dữ liệu: 60/20/20 theo tỷ lệ, không có ngày cố định.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Console Windows mặc định cp1252 -> vỡ khi in tiếng Việt. Ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.db import get_engine, init_db, utcnow  # noqa: E402
from model import ade_engine as E  # noqa: E402

DEFAULT_SERIES = ["coffee_robusta", "coffee_arabica", "cocoa", "rubber_tsr20"]
DEFAULT_HORIZONS = [1, 3, 6, 12]
DATA_START = "1990-01-01"      # hậu sụp đổ hạn ngạch ICA (7/1989) -> một cơ chế giá

BREAK_K = 2.0                  # quy tắc A
VOL_W, VOL_Q = 12, 0.75        # quy tắc B
PI_LEVEL = 0.80                # khoảng dự báo 80%

MODELS = ["ADE", "RW", "SAE", "ARIMA", "LSTM", "RF", "STACK"]
RIVALS = ["RW", "SAE", "ARIMA", "LSTM", "RF", "STACK"]   # đối chứng DM với ADE
DM_MIN_N = 10          # dưới ngưỡng này, DM vô nghĩa -> báo "chưa đủ quan sát"
REGIMES = ["full", "stable", "break", "vol_low", "vol_high"]


# ------------------------------------------------------------------- dữ liệu
def load_series(engine, series_id: str, start: str) -> pd.Series:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT obs_date, value FROM prices WHERE series_id=:s ORDER BY obs_date"),
            {"s": series_id},
        ).all()
    if not rows:
        raise RuntimeError(f"Chuỗi '{series_id}' chưa có dữ liệu. Chạy ingest trước.")
    s = pd.Series([float(r[1]) for r in rows],
                  index=pd.to_datetime([r[0] for r in rows]), name="y")
    s = s[s.index >= pd.Timestamp(start)]
    if s.empty:
        raise RuntimeError(f"Chuỗi '{series_id}' không có quan sát nào từ {start}.")
    return s


# ------------------------------------------------------------------- chế độ
def regime_masks(s_log: pd.Series, dates: pd.DatetimeIndex, train_end):
    """Hai quy tắc song song. Ngưỡng ước lượng CHỈ trên train."""
    dlog = s_log.diff()
    tr = dlog[dlog.index <= train_end]

    sigma = float(tr.std())
    thr_jump = BREAK_K * sigma
    jump = (dlog.reindex(dates).abs() > thr_jump).fillna(False).values

    rv = dlog.rolling(VOL_W).std()
    thr_vol = float(rv[rv.index <= train_end].quantile(VOL_Q))
    volhi = (rv.reindex(dates) > thr_vol).fillna(False).values

    masks = {
        "full":     np.ones(len(dates), dtype=bool),
        "break":    jump,
        "stable":   ~jump,
        "vol_high": volhi,
        "vol_low":  ~volhi,
    }
    return masks, {"jump": thr_jump, "vol": thr_vol, "sigma": sigma}


# ------------------------------------------------------------------- backtest
def run_one(series_id: str, s_native: pd.Series, horizon: int) -> dict:
    E.set_seeds()
    n = len(s_native)
    if n < 120:
        raise RuntimeError(f"{series_id}: chỉ {n} quan sát, cần ≥ 120.")

    use_log = bool((s_native > 0).all())
    s = np.log(s_native) if use_log else s_native.copy()
    back = np.exp if use_log else (lambda z: z)

    # chia 60/20/20 theo TỶ LỆ
    ts, vs = int(n * 0.60), int(n * 0.20)
    train_end, val_end = s.index[ts - 1], s.index[ts + vs - 1]

    arima_path, arima_model = E.fit_arima_wf(s.iloc[:ts], s.iloc[ts:], horizon)
    if arima_path is None:
        raise RuntimeError(f"{series_id}: ARIMA không hội tụ.")

    tgt = s.shift(-horizon)
    frame = E.build_frame(s, tgt, tgt - s, keep_future=True)
    members, y_val, y_test, dt, _ = E.ml_branch(frame, train_end, val_end, arima_path, s)

    ade_val, ade_test, _, W_te = E.build_ade(members, y_val, y_test, horizon)

    P = members["test"]
    preds = {
        "ADE":   ade_test,
        "ARIMA": P[:, 0],
        "LSTM":  P[:, 1],
        "RF":    P[:, 2],
        "SAE":   P.mean(axis=1),
        "STACK": E.stacking_offline(members["val"], y_val, P),
        "RW":    s.reindex(dt).values,          # baseline: giá tại thời điểm gốc
    }

    # khoảng dự báo từ sai số trên VALIDATION
    lo_off, hi_off = np.quantile(y_val - ade_val,
                                 [(1 - PI_LEVEL) / 2, 1 - (1 - PI_LEVEL) / 2])

    masks, thr = regime_masks(s, dt, train_end)

    return {
        "series_id": series_id, "horizon": horizon, "use_log": use_log, "back": back,
        "dates": dt, "y": back(y_test), "preds": {k: back(v) for k, v in preds.items()},
        "W": W_te, "masks": masks, "thr": thr,
        "lo_off": float(lo_off), "hi_off": float(hi_off),
        "train_end": train_end, "val_end": val_end,
        "n_test": len(dt),
        "s_log": s, "arima_model": arima_model, "frame": frame,
    }


# ------------------------------------------------------------------- dự báo tiến
def forecast_forward(rec: dict) -> dict | None:
    """Dự báo cho tháng chưa xảy ra. RF/LSTM huấn luyện lại trên toàn bộ dữ liệu
    có nhãn (khác backtest, vốn chỉ dùng train) — đúng cho sản phẩm thật, và
    được ghi rõ ở đây để không thành 'bí mật'."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    frame, h, s = rec["frame"], rec["horizon"], rec["s_log"]
    fut = frame[frame["target"].isna()]
    lab = frame[frame["target"].notna()]
    if fut.empty or len(rec["W"]) == 0:
        return None

    w = rec["W"][-1]
    anchor = float(s.iloc[-1])
    a_fut = rec["arima_model"].predict(h)

    rfc, lvc = E.cols_rf(frame), E.cols_lstm(frame)
    scr = MinMaxScaler()
    (rf_fu,) = E._fit_rf(scr.fit_transform(lab[rfc].values), lab["change"].values,
                         scr.transform(fut[rfc].values))
    seq = lambda d: d[lvc].values[:, ::-1]
    scl = StandardScaler()
    Xls = scl.fit_transform(seq(lab))
    yl = lab["change"].values
    n_es = max(int(len(lab) * 0.15), 12)
    (ls_fu,) = E._fit_lstm(Xls[:-n_es], yl[:-n_es], Xls[-n_es:], yl[-n_es:],
                           scl.transform(seq(fut)))

    point = float(np.array([a_fut, anchor + ls_fu[-1], anchor + rf_fu[-1]]) @ w)
    back = rec["back"]
    return {
        "run_date": s.index[-1].date(),
        "target_date": (s.index[-1] + pd.DateOffset(months=h)).date(),
        "point": float(back(point)),
        "lo": float(back(point + rec["lo_off"])),
        "hi": float(back(point + rec["hi_off"])),
    }


# ------------------------------------------------------------------- kiểm định
def dm_rows(rec: dict) -> list[dict]:
    """DM test: ADE so với từng đối thủ, tách theo chế độ.

    p_value = None khi n < DM_MIN_N. Đó KHÔNG phải lỗi — đó là câu trả lời:
    'chưa đủ quan sát để kết luận'. Câu đó phải được nói ra, không được lấp liếm.
    """
    out, y, h = [], rec["y"], rec["horizon"]
    for rg in REGIMES:
        m = rec["masks"][rg]
        n = int(m.sum())
        if n == 0:
            continue
        ya = y[m]
        pa = rec["preds"]["ADE"][m]
        ra = E.rmse(y, rec["preds"]["ADE"], m)
        for b in RIVALS:
            pb = rec["preds"][b][m]
            rb = E.rmse(y, rec["preds"][b], m)
            pv = E.dm_hln(ya, pa, pb, h) if n >= DM_MIN_N else np.nan
            out.append({
                "s": rec["series_id"], "h": h, "rg": rg, "a": "ADE", "b": b,
                "ra": float(ra), "rb": float(rb),
                "g": float(100 * (1 - ra / rb)) if rb else None,
                "p": None if (pv is None or np.isnan(pv)) else float(pv),
                "n": n,
            })
    return out


# ------------------------------------------------------------------- ghi CSDL
def persist(engine, rec: dict, fwd: dict | None) -> None:
    sid, h, now = rec["series_id"], rec["horizon"], utcnow()
    back, use_log = rec["back"], rec["use_log"]

    def band(p: float) -> tuple[float, float]:
        base = np.log(p) if use_log else p
        a, b = float(back(base + rec["lo_off"])), float(back(base + rec["hi_off"]))
        return min(a, b), max(a, b)

    fc, wt, mt = [], [], []
    for i, d in enumerate(rec["dates"]):
        run_d = d.date()
        tgt_d = (d + pd.DateOffset(months=h)).date()
        for m in MODELS:
            p = float(rec["preds"][m][i])
            lo, hi = band(p) if m == "ADE" else (p, p)   # chỉ ADE công bố khoảng
            fc.append({"s": sid, "rd": run_d, "td": tgt_d, "h": h, "p": p,
                       "lo": lo, "hi": hi, "m": m, "t": now})
        if h == 1:                                       # w(t) lưu một lần
            w = rec["W"][i]
            wt.append({"s": sid, "d": run_d, "a": float(w[0]),
                       "l": float(w[1]), "r": float(w[2]), "t": now})

    if fwd:
        fc.append({"s": sid, "rd": fwd["run_date"], "td": fwd["target_date"], "h": h,
                   "p": fwd["point"], "lo": fwd["lo"], "hi": fwd["hi"],
                   "m": "ADE", "t": now})

    for m in MODELS:
        for rg in REGIMES:
            mask = rec["masks"][rg]
            if mask.sum() == 0:
                continue
            mt.append({"s": sid, "m": f"{m}_h{h}", "rg": rg,
                       "rmse": E.rmse(rec["y"], rec["preds"][m], mask),
                       "mae": E.mae(rec["y"], rec["preds"][m], mask),
                       "n": int(mask.sum()), "t": now})

    dm = []
    for rec_dm in dm_rows(rec):
        rec_dm["t"] = now
        dm.append(rec_dm)

    ex = "EXCLUDED" if engine.dialect.name == "postgresql" else "excluded"
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO forecasts (series_id,run_date,target_date,horizon,point,lo,hi,model,created_at) "
            "VALUES (:s,:rd,:td,:h,:p,:lo,:hi,:m,:t) "
            f"ON CONFLICT (series_id,run_date,target_date,model) DO UPDATE SET "
            f"point={ex}.point, lo={ex}.lo, hi={ex}.hi, horizon={ex}.horizon, created_at={ex}.created_at"
        ), fc)
        if wt:
            conn.execute(text(
                "INSERT INTO weights (series_id,obs_date,w_arima,w_lstm,w_rf,created_at) "
                "VALUES (:s,:d,:a,:l,:r,:t) "
                f"ON CONFLICT (series_id,obs_date) DO UPDATE SET w_arima={ex}.w_arima, "
                f"w_lstm={ex}.w_lstm, w_rf={ex}.w_rf, created_at={ex}.created_at"
            ), wt)
        conn.execute(text(
            "INSERT INTO metrics (series_id,model,regime,rmse,mae,n_obs,created_at) "
            "VALUES (:s,:m,:rg,:rmse,:mae,:n,:t) "
            f"ON CONFLICT (series_id,model,regime) DO UPDATE SET rmse={ex}.rmse, "
            f"mae={ex}.mae, n_obs={ex}.n_obs, created_at={ex}.created_at"
        ), mt)
        if dm:
            conn.execute(text(
                "INSERT INTO dm_tests (series_id,horizon,regime,model_a,model_b,"
                "rmse_a,rmse_b,gain_pct,p_value,n_obs,created_at) "
                "VALUES (:s,:h,:rg,:a,:b,:ra,:rb,:g,:p,:n,:t) "
                f"ON CONFLICT (series_id,horizon,regime,model_b) DO UPDATE SET "
                f"rmse_a={ex}.rmse_a, rmse_b={ex}.rmse_b, gain_pct={ex}.gain_pct, "
                f"p_value={ex}.p_value, n_obs={ex}.n_obs, created_at={ex}.created_at"
            ), dm)


# ------------------------------------------------------------------- báo cáo
def report(rec: dict, fwd: dict | None) -> None:
    y, m, h, thr = rec["y"], rec["masks"], rec["horizon"], rec["thr"]

    def cell(model, rg):
        v = E.rmse(y, rec["preds"][model], m[rg])
        return f"{v:9.4f}" if not np.isnan(v) else "      n/a"

    print(f"   h={h:<2d} train→{rec['train_end'].date()} | val→{rec['val_end'].date()} | "
          f"test {rec['dates'][0].date()}→{rec['dates'][-1].date()}  (n={rec['n_test']})")
    print(f"        A · cú nhảy       |Δlog| > {thr['jump']:.4f}"
          f"   → {int(m['break'].sum()):3d}/{rec['n_test']} tháng")
    print(f"        B · biến động cao  σ₁₂  > {thr['vol']:.4f}"
          f"   → {int(m['vol_high'].sum()):3d}/{rec['n_test']} tháng")
    print(f"        RMSE            ADE       RW      SAE    ARIMA     LSTM       RF")
    for rg in REGIMES:
        if m[rg].sum() == 0:
            continue
        print(f"        {rg:10s}" + "".join(cell(k, rg)
              for k in ("ADE", "RW", "SAE", "ARIMA", "LSTM", "RF")))
    print(f"        DM (ADE vs …)     gain%     p-value    kết luận")
    for rg in ("full", "break", "vol_high"):
        if m[rg].sum() == 0:
            continue
        for row in [r for r in dm_rows(rec) if r["rg"] == rg and r["b"] in ("RW", "SAE")]:
            if row["p"] is None:
                verdict = f"n={row['n']} < {DM_MIN_N}: chưa đủ quan sát"
                ptxt = "     —  "
            elif row["p"] < 0.05:
                verdict = "CÓ ý nghĩa (p<0.05)" if row["g"] > 0 else "THUA có ý nghĩa"
                ptxt = f"{row['p']:8.4f}"
            else:
                verdict = "chưa có ý nghĩa thống kê"
                ptxt = f"{row['p']:8.4f}"
            if row['g'] is None:
                gtxt = "   n/a   "
            else:
                try:
                    gtxt = f"{row['g']:7.1f}%"
                except Exception as e:
                    print(f"DEBUG: rg={rg}, b={row['b']}, g={repr(row['g'])}, type={type(row['g'])}")
                    raise
            print(f"        {rg:9s} vs {row['b']:5s}{gtxt}  {ptxt}   {verdict}")
    if fwd:
        print(f"        → dự báo {fwd['target_date']}: {fwd['point']:.3f} "
              f"[{fwd['lo']:.3f} – {fwd['hi']:.3f}]")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=DEFAULT_SERIES)
    ap.add_argument("--horizons", nargs="*", type=int, default=DEFAULT_HORIZONS)
    ap.add_argument("--start", default=DATA_START, help="ngày bắt đầu chuỗi (YYYY-MM-DD)")
    ap.add_argument("--db")
    args = ap.parse_args()

    engine = get_engine(args.db)
    init_db(engine)

    print(f"ARIMA: {'pmdarima' if E.HAS_PM else 'statsmodels (fallback)'} | "
          f"LSTM: {'TensorFlow' if E.HAS_TF else 'MLP (fallback)'}\n")

    for sid in args.series:
        s = load_series(engine, sid, args.start)
        print(f"── {sid}: {len(s)} quan sát [{s.index[0].date()} → {s.index[-1].date()}]")
        for h in args.horizons:
            rec = run_one(sid, s, h)
            fwd = forecast_forward(rec)
            persist(engine, rec, fwd)
            report(rec, fwd)

    print("Đã ghi vào bảng forecasts / weights / metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
