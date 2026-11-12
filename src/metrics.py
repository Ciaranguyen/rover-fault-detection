"""Thước đo dùng chung cho mọi bộ phát hiện.

Đầu vào chuẩn cho mọi mô hình:
    prob  : (M, 3) xác suất 3 lớp [trung thực, trôi, khai giả] cho từng mẫu (episode, bước, rover)
    y     : (M,)   nhãn thật
    idx   : (M, 3) chỉ số (episode, bước, rover) của từng mẫu
Điểm "có lỗi" = 1 - prob[:, 0].
"""
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def threshold_at_fpr(score, y, fpr=0.01):
    """Ngưỡng nhỏ nhất sao cho tỉ lệ báo nhầm trên mẫu trung thực <= fpr. Chỉ gọi trên tập val."""
    s_honest = np.sort(score[y == 0])
    k = int(np.ceil((1 - fpr) * len(s_honest))) - 1
    return float(s_honest[min(max(k, 0), len(s_honest) - 1)])


def detection_delay(alarm, y, idx, meta_episodes):
    """Với mỗi rover lỗi: số bước từ khi lỗi bắt đầu tới lần báo động đầu tiên (sau t_u).
    Trả về độ trễ trung vị (bước) và tỉ lệ bỏ sót (không báo động lần nào tới hết episode)."""
    delays, missed = [], 0
    fault_rows = np.where(y > 0)[0]
    if len(fault_rows) == 0:
        return float("nan"), float("nan")
    keys = {}
    for r in fault_rows:
        keys.setdefault((idx[r, 0], idx[r, 2]), []).append(r)
    for (e, n), rows in keys.items():
        rows = np.asarray(rows)
        rows = rows[np.argsort(idx[rows, 1])]
        t_u = meta_episodes[e]["t_u"]
        hit = np.where(alarm[rows])[0]
        if len(hit) == 0:
            missed += 1
        else:
            delays.append(idx[rows[hit[0]], 1] - t_u)
    n_f = len(keys)
    return (float(np.median(delays)) if delays else float("nan")), missed / n_f


def ece(prob_fault, is_fault, bins=15):
    """Expected calibration error cho xác suất "có lỗi"."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(prob_fault, edges) - 1, 0, bins - 1)
    total, err = len(prob_fault), 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            err += m.sum() / total * abs(prob_fault[m].mean() - is_fault[m].mean())
    return float(err)


def evaluate(prob, y, idx, meta_episodes, thr, score=None, calibrated=True):
    """thr: ngưỡng điểm lỗi đã chọn trên tập val (báo nhầm <= 1%).
    score: điểm lỗi riêng (mặc định 1 - prob[:, 0]); calibrated=False nếu mô hình không ra xác suất."""
    pred = prob.argmax(1)
    score = 1 - prob[:, 0] if score is None else score
    alarm = score >= thr
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    honest = y == 0
    delay, miss = detection_delay(alarm, y, idx, meta_episodes)
    out = {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "precision": {"honest": float(p[0]), "drift": float(p[1]), "spoof": float(p[2])},
        "recall": {"honest": float(r[0]), "drift": float(r[1]), "spoof": float(r[2])},
        "threshold": thr,
        "fpr_at_thr": float(alarm[honest].mean()),
        "recall_fault_at_thr": float(alarm[~honest].mean()),
        "recall_drift_at_thr": float(alarm[y == 1].mean()) if (y == 1).any() else float("nan"),
        "recall_spoof_at_thr": float(alarm[y == 2].mean()) if (y == 2).any() else float("nan"),
        "median_delay_steps": delay,
        "missed_faulty_rovers": miss,
        "ece_fault": ece(score, (~honest).astype(float)) if calibrated else float("nan"),
        "n_samples": int(len(y)),
    }
    return out
