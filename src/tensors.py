"""Chuyển một tập dữ liệu thành tensor cho các mô hình nơ-ron.

Quy ước chỉ số cạnh: [..., i, k] = rover i (người quan sát) thấy rover k (đối tượng cần phân loại).
Mọi đầu vào chỉ lấy từ p_hat và r (những gì rover thật sự có); không dùng p_true hay label.

    edge   (E,T,N,N,3)  [e_ik, |e_ik|, có đo hay không]   (NaN -> 0)
    valid  (E,T,N,N)    có phép đo ở bước t (i != k)
    recent (E,T,N,N)    có ít nhất một phép đo trong `hist` bước gần nhất
    node   (E,T,N,6)    thống kê bằng chứng về rover k ở bước t:
                        mean|e|, max|e|, min|e|, mean e, số người quan sát, tốc độ khai của k
    label  (E,T,N)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import residuals  # noqa: E402


def build(split, data_dir="data", hist=30):
    d = np.load(f"{data_dir}/{split}.npz")
    meta = json.load(open(f"{data_dir}/{split}_meta.json"))["episodes"]
    e = residuals(d).astype(np.float32)                   # (E,T,N,N)
    E, T, N, _ = e.shape
    valid = np.isfinite(e) & ~np.eye(N, dtype=bool)[None, None]
    e0 = np.where(valid, e, 0.0).astype(np.float32)
    a0 = np.abs(e0)
    edge = np.stack([e0, a0, valid.astype(np.float32)], -1)

    cs = np.cumsum(valid, axis=1, dtype=np.int32)
    prev = np.concatenate([np.zeros((E, hist, N, N), np.int32), cs[:, :-hist]], axis=1)
    recent = (cs - prev) > 0

    cnt = valid.sum(axis=2).astype(np.float32)            # gộp theo người quan sát i
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ea = np.where(valid, np.abs(e), np.nan)
        es = np.where(valid, e, np.nan)
        stats = [np.nanmean(ea, 2), np.nanmax(ea, 2), np.nanmin(ea, 2), np.nanmean(es, 2)]
    ph = d["p_hat"]
    speed = np.concatenate([np.zeros((E, 1, N), np.float32),
                            np.linalg.norm(np.diff(ph, axis=1), axis=-1)], axis=1)
    node = np.stack([np.nan_to_num(s, nan=0.0) for s in stats] + [cnt, speed], -1).astype(np.float32)
    return {"edge": edge, "valid": valid, "recent": recent, "node": node,
            "label": d["label"].astype(np.int64), "meta": meta}
