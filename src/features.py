"""Tính bằng chứng và đặc trưng từ dữ liệu sinh ra.

Bằng chứng gốc: phần dư range mà rover i thấy về rover k tại bước t
    e_ik(t) = r_ik(t) - ||p_hat_i(t) - p_hat_k(t)||      (NaN nếu không đo được)
Mô hình KHÔNG được dùng p_true hay label làm đầu vào.

Đặc trưng thủ công cho baseline (ngưỡng, hồi quy logistic): với mỗi rover k tại bước t,
tóm tắt bằng chứng trong cửa sổ W bước gần nhất (chỉ dùng quá khứ, tức nhân quả).
"""
import warnings

import numpy as np

FEATURE_NAMES = [
    "win_mean_abs",      # trung bình theo cửa sổ của mean_i |e_ik|   (điểm của baseline ngưỡng)
    "win_max_abs",       # ... của max_i |e_ik|
    "win_min_abs",       # ... của min_i |e_ik|
    "win_med_abs",       # ... của median_i |e_ik|
    "win_spread",        # ... của (max - min)_i |e_ik|  (lớn khi chỉ một số hàng xóm thấy lệch)
    "win_signed",        # ... của mean_i e_ik (có dấu)
    "last_mean_abs",     # mean_i |e_ik| ở bước hiện tại
    "trend",             # nửa sau cửa sổ trừ nửa đầu (trôi thì tăng dần)
    "win_neighbors",     # số hàng xóm đo được trung bình
    "win_speed",         # tốc độ khai trung bình |p_hat_k(t) - p_hat_k(t-1)|
    "win_speed_std",     # độ lệch chuẩn tốc độ khai
    "no_evidence",       # 1 nếu cả cửa sổ không có phép đo nào
]


def residuals(data):
    """data: dict/npz có p_hat (E,T,N,2), r (E,T,N,N). Trả về e (E,T,N,N), e[..., i, k]."""
    ph = data["p_hat"].astype(np.float32)
    est = np.linalg.norm(ph[:, :, :, None, :] - ph[:, :, None, :, :], axis=-1)
    return data["r"] - est


def _rolling_mean(x, w):
    """Trung bình trượt nhân quả theo trục 1 (thời gian), bỏ qua NaN. x: (E,T,...)"""
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0).astype(np.float64)
    c = valid.astype(np.float64)
    cv, cc = np.cumsum(v, axis=1), np.cumsum(c, axis=1)
    pad = [(0, 0)] * x.ndim
    pad[1] = (w, 0)
    cv = np.pad(cv, pad)[:, :-w] if w > 0 else cv
    cc = np.pad(cc, pad)[:, :-w] if w > 0 else cc
    sv = np.cumsum(v, axis=1) - cv
    sc = np.cumsum(c, axis=1) - cc
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sv / sc).astype(np.float32), sc.astype(np.float32)


def window_features(data, w=30):
    """Đặc trưng (E,T,N,F). Các bước t < w-1 có cửa sổ chưa đủ, nên bị loại khi đánh giá."""
    e = residuals(data)                       # (E,T,N,N): [.., i, k] = i thấy k
    a = np.abs(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if True:
            m_mean = np.nanmean(a, axis=2)    # (E,T,N): gộp theo người quan sát i
            m_max = np.nanmax(a, axis=2)
            m_min = np.nanmin(a, axis=2)
            m_med = np.nanmedian(a, axis=2)
            m_sgn = np.nanmean(e, axis=2)
    nnb = np.isfinite(e).sum(axis=2).astype(np.float32)
    ph = data["p_hat"]
    speed = np.concatenate([np.full(ph.shape[:1] + (1,) + ph.shape[2:3], np.nan, np.float32),
                            np.linalg.norm(np.diff(ph, axis=1), axis=-1)], axis=1).astype(np.float32)

    f_mean, cnt = _rolling_mean(m_mean, w)
    f_max, _ = _rolling_mean(m_max, w)
    f_min, _ = _rolling_mean(m_min, w)
    f_med, _ = _rolling_mean(m_med, w)
    f_spread, _ = _rolling_mean(m_max - m_min, w)
    f_sgn, _ = _rolling_mean(m_sgn, w)
    h = max(w // 2, 1)
    first, _ = _rolling_mean(np.roll(m_mean, h, axis=1), w - h)   # nửa đầu cửa sổ
    second, _ = _rolling_mean(m_mean, h)                            # nửa sau cửa sổ
    trend = second - first
    f_nnb, _ = _rolling_mean(nnb, w)
    f_spd, _ = _rolling_mean(speed, w)
    f_spd2, _ = _rolling_mean(speed ** 2, w)
    f_spd_std = np.sqrt(np.maximum(f_spd2 - f_spd ** 2, 0))
    no_ev = (cnt == 0).astype(np.float32)

    X = np.stack([f_mean, f_max, f_min, f_med, f_spread, f_sgn, m_mean, trend,
                  f_nnb, f_spd, f_spd_std, no_ev], axis=-1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def flatten(X, y, w, stride=1):
    """Chuyển (E,T,N,F) thành mẫu (M,F), bỏ các bước cửa sổ chưa đủ; trả kèm chỉ số (e,t,n)."""
    E, T, N, F = X.shape
    ts = np.arange(w - 1, T, stride)
    Xs, ys = X[:, ts], y[:, ts]
    ee, tt, nn = np.meshgrid(np.arange(E), ts, np.arange(N), indexing="ij")
    return Xs.reshape(-1, F), ys.reshape(-1), np.stack([ee, tt, nn], -1).reshape(-1, 3)
