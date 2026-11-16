"""E8: gộp kết quả mọi seed (cả Pro và Free), tính so sánh ghép cặp và kết luận cho H1–H3.

Mọi so sánh đều GHÉP CẶP: cùng seed chính sách, cùng episode, chỉ khác nguồn tin cậy / chính sách.
KTC 95%: bootstrap hai tầng (lấy mẫu lại seed và episode), 2000 lần. Kèm IQM qua seed (Agarwal et al., 2021).
Đơn vị: holes và độ phủ tính theo tỉ lệ diện tích; 1 điểm % = 5,76 m² (vùng 24 m x 24 m).

Chạy:  python src/e8_analyze.py
Ghi:   results/runs_<ACC>/e8/E8_summary.json
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

AREA = 24.0 * 24.0
CONTRASTS = [  # (a, b, thước đo, giả thuyết)
    ("none", "oracle", "holes", "H1"), ("logistic", "oracle", "holes", "H1"), ("gru", "oracle", "holes", "H1"),
    ("grugat", "oracle", "holes", "H1"), ("none", "logistic", "holes", "H1"), ("logistic", "gru", "holes", "H1"),
    ("gru", "grugat", "holes", "H1"),
    ("grugat", "gru", "coverage", "H2"), ("grugat+blame0.7%", "grugat", "coverage", "H2"),
    ("grugat+blame2.1%", "grugat", "coverage", "H2"), ("grugat+blame10%", "grugat", "coverage", "H2"),
    ("grugat+blame10%", "grugat", "holes", "H2"), ("oracle", "none", "coverage", "H2"),
    ("oracle", "heur_oracle", "coverage", "H3"), ("grugat", "heur_grugat", "coverage", "H3"),
]


def load():
    files = sorted(glob.glob("results/runs_*/e8/eval_s*.json"))
    runs = [json.load(open(f)) for f in files]
    seeds = [r["seed"] for r in runs]
    conds = [c for c in runs[0]["conditions"] if all(c in r["conditions"] for r in runs)]
    X = {m: {c: np.array([r["conditions"][c][m] for r in runs], float) for c in conds}
         for m in ["coverage", "holes", "false_flag_rate"]}
    fault = np.array(runs[0]["conditions"][conds[0]]["has_fault"], bool)
    ftype = np.array(runs[0]["conditions"][conds[0]]["ftype"])
    return seeds, conds, X, fault, ftype


def iqm(v):
    v = np.sort(v)
    k = len(v) // 4
    return float(v[k:len(v) - k].mean())


def boot(D, B=2000, seed=0):
    """D (S,E) hiệu số ghép cặp. Trả về mẫu bootstrap (B,) của trung bình, lấy lại cả seed lẫn episode."""
    rng = np.random.default_rng(seed)
    S, E = D.shape
    W = np.stack([np.bincount(rng.integers(0, E, E), minlength=E) for _ in range(B)]).astype(float)   # (B,E)
    per = (D @ W.T) / W.sum(1)                                                                       # (S,B)
    pick = rng.integers(0, S, (B, S))
    return per[pick, np.arange(B)[:, None]].mean(1)


def main():
    acc = os.environ.get("ACC", "local")
    seeds, conds, X, fault, ftype = load()
    S, E = X["coverage"][conds[0]].shape
    out = {"seeds": seeds, "n_episodes": E, "n_fault_episodes": int(fault.sum()), "conditions": {}, "contrasts": []}
    print(f"{S} seed chính sách × {E} episode ({fault.sum()} có lỗi)\n")
    print(f"{'điều kiện':18s} {'holes(lỗi) TB [IQM]':>24s} {'phủ(lỗi) TB [IQM]':>22s} {'phủ(không lỗi)':>15s} {'báo nhầm/mốc':>13s}")
    for c in conds:
        h, cv = X["holes"][c], X["coverage"][c]
        row = {"holes_fault": float(h[:, fault].mean()), "holes_fault_iqm": iqm(h[:, fault].mean(1)),
               "cov_fault": float(cv[:, fault].mean()), "cov_fault_iqm": iqm(cv[:, fault].mean(1)),
               "cov_nofault": float(cv[:, ~fault].mean()), "false_flag_rate": float(X["false_flag_rate"][c].mean()),
               "holes_by_type": {t: float(h[:, ftype == t].mean()) for t in ["drift", "spoof", "spoof_one"]},
               "per_seed_holes_fault": h[:, fault].mean(1).tolist(), "per_seed_cov_fault": cv[:, fault].mean(1).tolist()}
        out["conditions"][c] = row
        print(f"{c:18s} {row['holes_fault']*100:8.3f}% [{row['holes_fault_iqm']*100:6.3f}%]   "
              f"{row['cov_fault']*100:7.2f}% [{row['cov_fault_iqm']*100:6.2f}%]   {row['cov_nofault']*100:12.2f}%   "
              f"{row['false_flag_rate']:12.4f}")
    print("\nSo sánh ghép cặp trên episode có lỗi (điểm %, KTC 95% hai tầng; dương = a lớn hơn b):")
    for a, b, m, hyp in CONTRASTS:
        if a not in conds or b not in conds:
            continue
        D = (X[m][a] - X[m][b])[:, fault] * 100
        bs = boot(D)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rec = {"a": a, "b": b, "metric": m, "hyp": hyp, "mean_pp": float(D.mean()), "ci_pp": [float(lo), float(hi)],
               "mean_m2": float(D.mean() / 100 * AREA), "iqm_pp": iqm(D.mean(1)),
               "seeds_positive": int((D.mean(1) > 0).sum())}
        out["contrasts"].append(rec)
        sig = "có ý nghĩa" if lo > 0 or hi < 0 else "không phân biệt được"
        print(f"  [{hyp}] {m:8s} {a:18s} − {b:14s} {rec['mean_pp']:+7.3f} [{lo:+7.3f}, {hi:+7.3f}]  "
              f"({rec['mean_m2']:+6.2f} m²; {rec['seeds_positive']}/{S} seed dương)  {sig}")
    path = Path(f"results/runs_{acc}/e8/E8_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
