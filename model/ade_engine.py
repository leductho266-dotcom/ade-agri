"""ADE-NoSTL engine — trích từ pipeline luận án (v8c), giữ nguyên lõi toán học.

Thay đổi duy nhất so với bản gốc:
  * bỏ phần Google Drive / FRED (dữ liệu nay lấy từ CSDL);
  * bổ sung fallback cho pmdarima và TensorFlow để chạy được ở mọi môi trường;
  * fit_arima_wf trả thêm mô hình đã fit (phục vụ dự báo tiến).
Toán của ade_slsqp, ml_branch, build_frame KHÔNG bị sửa.
"""
from __future__ import annotations

import os
import random
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
N_LAGS_RF, LSTM_SEQ = 6, 18
ROLL_MEAN_W, ROLL_STD_W = 3, 6
L_SLSQP, LSTM_UNITS = 6, 16

try:
    import pmdarima as pm
    HAS_PM = True
except Exception:
    HAS_PM = False
    from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    HAS_TF = True
except Exception:
    HAS_TF = False
    from sklearn.neural_network import MLPRegressor


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TF:
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass


# ---------------------------------------------------------------- ARIMA
class _ArimaWrap:
    """Giao diện thống nhất cho pmdarima và statsmodels."""

    def __init__(self, hist):
        self.hist = list(hist)
        self._fit()

    def _fit(self):
        if HAS_PM:
            self.m = pm.auto_arima(np.asarray(self.hist), seasonal=False, stepwise=True,
                                   suppress_warnings=True, error_action="ignore")
        else:
            best, best_aic = None, np.inf
            for order in [(1, 1, 1), (2, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 2)]:
                try:
                    r = SM_ARIMA(np.asarray(self.hist), order=order).fit()
                    if r.aic < best_aic:
                        best, best_aic = r, r.aic
                except Exception:
                    continue
            self.m = best

    def update(self, value):
        self.hist.append(value)
        if HAS_PM:
            try:
                self.m.update(value)
                return
            except Exception:
                pass
        self._fit()

    def predict(self, h: int) -> float:
        try:
            if HAS_PM:
                fc = self.m.predict(n_periods=h)
                return float(np.asarray(fc)[-1])
            return float(np.asarray(self.m.forecast(steps=h))[-1])
        except Exception:
            return float(self.hist[-1])


def fit_arima_wf(tr: pd.Series, vt: pd.Series, horizon: int, refit_every: int = 24):
    """Walk-forward ARIMA. Trả (chuỗi dự báo trên vt, mô hình cuối cùng)."""
    try:
        model = _ArimaWrap(tr.values)
    except Exception:
        return None, None
    out = np.full(len(vt), np.nan)
    for i in range(len(vt)):
        model.update(vt.iloc[i])
        if (i + 1) % refit_every == 0:
            model._fit()
        out[i] = model.predict(horizon)
    return pd.Series(out, index=vt.index), model


# ---------------------------------------------------------------- ML members
def _fit_rf(Xtr, ytr, *blocks):
    sc = StandardScaler()
    ys = sc.fit_transform(ytr.reshape(-1, 1)).ravel()
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr, ys)
    return [sc.inverse_transform(rf.predict(B).reshape(-1, 1)).ravel() if len(B) else np.array([])
            for B in blocks]


def _fit_lstm(Xtr, ytr, Xes, yes, *blocks):
    sc = StandardScaler()
    ys = sc.fit_transform(ytr.reshape(-1, 1)).ravel()
    if not HAS_TF:
        mlp = MLPRegressor(hidden_layer_sizes=(LSTM_UNITS,), max_iter=400, random_state=SEED,
                           early_stopping=True, validation_fraction=0.15)
        mlp.fit(Xtr, ys)
        return [sc.inverse_transform(mlp.predict(B).reshape(-1, 1)).ravel() if len(B) else np.array([])
                for B in blocks]
    set_seeds()
    yes_s = sc.transform(yes.reshape(-1, 1)).ravel()
    r = lambda X: X.reshape(X.shape[0], X.shape[1], 1)
    m = Sequential([LSTM(LSTM_UNITS, input_shape=(Xtr.shape[1], 1)), Dropout(0.1), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    m.fit(r(Xtr), ys, validation_data=(r(Xes), yes_s), epochs=80, batch_size=16,
          verbose=0, callbacks=[es])
    return [sc.inverse_transform(m.predict(r(B), verbose=0)).ravel() if len(B) else np.array([])
            for B in blocks]


def build_frame(feat: pd.Series, target: pd.Series, change: pd.Series,
                lstm_k: int = LSTM_SEQ, rf_k: int = N_LAGS_RF, keep_future: bool = False):
    df = pd.DataFrame(index=feat.index)
    for L in range(1, lstm_k + 1):
        df[f"lev_{L}"] = feat.shift(L - 1)
    d = feat.diff()
    for L in range(1, rf_k + 1):
        df[f"dif_{L}"] = d.shift(L - 1)
    df["rmean"] = feat.rolling(ROLL_MEAN_W).mean()
    df["rstd"] = feat.rolling(ROLL_STD_W).std()
    mo = pd.get_dummies(df.index.month, prefix="mo")
    mo.index = df.index
    for c in mo.columns[1:]:
        df[c] = mo[c].astype(float)
    df["target"] = target
    df["change"] = change
    feat_cols = [c for c in df.columns if c not in ("target", "change")]
    df = df.dropna(subset=feat_cols)
    return df if keep_future else df.dropna()


def cols_lstm(df):
    return [c for c in df.columns if c.startswith("lev_")]


def cols_rf(df):
    return [c for c in df.columns if c.startswith(("dif_", "rmean", "rstd", "mo"))]


def ml_branch(frame, train_end, val_end, arima_level, recon_anchor):
    """Giữ nguyên logic gốc. frame có thể chứa các dòng future (target = NaN)."""
    has_t = frame["target"].notna()
    tr = frame[(frame.index <= train_end) & has_t]
    va = frame[(frame.index > train_end) & (frame.index <= val_end) & has_t]
    te = frame[(frame.index > val_end) & has_t]
    fu = frame[~has_t]

    n_es = max(int(len(tr) * 0.15), 12)
    ptr, pes = tr.iloc[:-n_es], tr.iloc[-n_es:]
    ytr, yes = ptr["change"].values, pes["change"].values
    rfc, lvc = cols_rf(frame), cols_lstm(frame)

    scr = MinMaxScaler()
    Xtr_rf = scr.fit_transform(ptr[rfc].values)
    rf_va, rf_te, rf_fu = _fit_rf(Xtr_rf, ytr, scr.transform(va[rfc].values),
                                  scr.transform(te[rfc].values),
                                  scr.transform(fu[rfc].values) if len(fu) else np.empty((0, len(rfc))))

    seq = lambda d: d[lvc].values[:, ::-1]
    scl = StandardScaler()
    Xtr_ls = scl.fit_transform(seq(ptr))
    ls_va, ls_te, ls_fu = _fit_lstm(Xtr_ls, ytr, scl.transform(seq(pes)), yes,
                                    scl.transform(seq(va)), scl.transform(seq(te)),
                                    scl.transform(seq(fu)) if len(fu) else np.empty((0, len(lvc))))

    def stack(idx, ls, rf):
        a = arima_level.reindex(idx).values
        anc = recon_anchor.reindex(idx).values
        return np.column_stack([a, anc + ls, anc + rf])

    members = {"val": stack(va.index, ls_va, rf_va), "test": stack(te.index, ls_te, rf_te)}
    return members, va["target"].values, te["target"].values, te.index, va.index


def ade_slsqp(y_full, P_full, horizon, L: int = L_SLSQP):
    """Meta-learner SLSQP — KHÔNG sửa so với bản gốc."""
    n, m = P_full.shape
    w = np.ones(m) / m
    W = np.zeros((n, m))
    pred = np.zeros(n)
    cons = ({"type": "eq", "fun": lambda v: np.sum(v) - 1.0},)
    bnds = [(0.0, 1.0)] * m
    loss = lambda v, P, Y: np.mean((Y - P @ v) ** 2)
    start = L + horizon - 1
    for i in range(n):
        if i >= start:
            e = i - horizon + 1
            s = e - L
            Pw, Yw = P_full[s:e], y_full[s:e]
            if not (np.isnan(Pw).any() or np.isnan(Yw).any()):
                try:
                    r = minimize(loss, w, args=(Pw, Yw), method="SLSQP", bounds=bnds,
                                 constraints=cons, options={"maxiter": 60, "ftol": 1e-7})
                    if r.success:
                        w = r.x
                except Exception:
                    pass
        W[i] = w
        pred[i] = P_full[i] @ w
    return pred, W


def build_ade(members, y_val, y_test, horizon):
    """Trả dự báo ADE trên cả val và test, kèm quỹ đạo trọng số."""
    vl = len(y_val)
    pred, W = ade_slsqp(np.concatenate([y_val, y_test]),
                        np.vstack([members["val"], members["test"]]), horizon)
    return pred[:vl], pred[vl:], W[:vl], W[vl:]


def stacking_offline(P_val, y_val, P_test):
    st = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=SEED)
    st.fit(P_val, y_val)
    return st.predict(P_test)


def dm_hln(y, p1, p2, horizon, power: int = 2):
    """Diebold–Mariano, hiệu chỉnh Harvey–Leybourne–Newbold."""
    d = np.abs(y - p1) ** power - np.abs(y - p2) ** power
    n = len(d)
    if n < 4:
        return np.nan
    g = np.var(d, ddof=0)
    for k in range(1, horizon):
        if k < n:
            g += 2 * np.cov(d[k:], d[:-k])[0, 1]
    vd = g / n
    if vd <= 0:
        return np.nan
    dm = d.mean() / np.sqrt(vd)
    corr = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    return 2 * sps.t.cdf(-abs(dm * np.sqrt(max(corr, 1e-8))), df=n - 1)


def rmse(y, p, m):
    return float(np.sqrt(mean_squared_error(y[m], p[m]))) if m.sum() else np.nan


def mae(y, p, m):
    return float(mean_absolute_error(y[m], p[m])) if m.sum() else np.nan
