"""E1: hai baseline không dùng mạng nơ-ron.

  threshold : điểm = trung bình theo cửa sổ của mean_i |e_ik| (giống lớp kiểm range trong proposal).
              Hai ngưỡng: s <= t1 -> trung thực; t1 < s <= t2 -> trôi; s > t2 -> khai giả.
              t1 chọn sao cho báo nhầm <= 1% trên val; t2 chọn để macro-F1 trên val cao nhất.
  logistic  : hồi quy logistic đa lớp trên 12 đặc trưng cửa sổ (src/features.py),
              chuẩn hoá đặc trưng, trọng số lớp cân bằng; C chọn theo macro-F1 trên val.

Quy trình đúng:
  1) Chế độ dò (mặc định): fit trên train, chọn cấu hình trên val, ghi results/.../E1_<model>_val.json
  2) Chế độ cuối (--final, chỉ chạy MỘT lần vào ngày đánh giá): dùng cấu hình đã chọn,
     đánh giá trên test và các tập ood_*, ghi E1_<model>_final.json

Chạy:
  python src/baselines.py --model threshold
  python src/baselines.py --model logistic
  python src/baselines.py --model logistic --final --splits test ood_drift8 ood_noise2 ood_two_faulty ood_n8
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_NAMES, flatten, window_features  # noqa: E402
from metrics import evaluate, threshold_at_fpr  # noqa: E402


def load(split, data_dir="data"):
    d = dict(np.load(f"{data_dir}/{split}.npz"))
    meta = json.load(open(f"{data_dir}/{split}_meta.json"))["episodes"]
    return d, meta


def prepare(split, w, stride=1, data_dir="data"):
    d, meta = load(split, data_dir)
    X = window_features(d, w)
    Xf, yf, idx = flatten(X, d["label"], w, stride)
    return Xf, yf.astype(int), idx, meta


def onehot(c, k=3):
    out = np.zeros((len(c), k), np.float32)
    out[np.arange(len(c)), c] = 1
    return out


# ---------------- threshold
def threshold_predict(s, t1, t2):
    c = np.zeros(len(s), int)
    c[s > t1] = 1
    c[s > t2] = 2
    return c


def tune_threshold(Xv, yv):
    s = Xv[:, 0]
    t1 = threshold_at_fpr(s, yv, 0.01)
    cands = np.quantile(s[s > t1], np.linspace(0.02, 0.98, 49)) if (s > t1).any() else [t1]
    best = max(((f1_score(yv, threshold_predict(s, t1, t2), average="macro", labels=[0, 1, 2],
                          zero_division=0), float(t2)) for t2 in cands))
    return {"t1": t1, "t2": best[1], "val_macro_f1_for_t2": best[0]}


# ---------------- logistic
def fit_logistic(Xtr, ytr, C):
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        C=C, class_weight="balanced", max_iter=2000))
    return clf.fit(Xtr, ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["threshold", "logistic"], required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--train_stride", type=int, default=5, help="lấy mẫu thưa trên train cho nhẹ")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--final", action="store_true", help="đánh giá cuối trên test/ood (chạy một lần)")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--tag", default="", help="hậu tố tên file kết quả, ví dụ _gauss")
    args = ap.parse_args()

    acc = os.environ.get("ACC", "local")
    out_dir = Path(f"results/runs_{acc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("checkpoints")
    model_dir.mkdir(exist_ok=True)
    name = f"E1_{args.model}_w{args.window}{args.tag}"
    t0 = time.time()

    Xv, yv, iv, mv = prepare(args.val_split, args.window)
    if args.model == "threshold":
        cfg = tune_threshold(Xv, yv)
        predict = lambda X: (onehot(threshold_predict(X[:, 0], cfg["t1"], cfg["t2"])), X[:, 0])
        thr = cfg["t1"]
        calibrated = False
        grid = None
    else:
        Xtr, ytr, _, _ = prepare(args.train_split, args.window, args.train_stride)
        grid = {}
        best = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = fit_logistic(Xtr, ytr, C)
            f = f1_score(yv, clf.predict(Xv), average="macro", labels=[0, 1, 2], zero_division=0)
            grid[str(C)] = float(f)
            print(f"  C={C}: val macro-F1 = {f:.4f}")
            if best is None or f > best[0]:
                best = (f, C, clf)
        clf = best[2]
        cfg = {"C": best[1], "n_train_samples": int(len(ytr))}
        predict = lambda X: (clf.predict_proba(X).astype(np.float32), None)
        pv, _ = predict(Xv)
        thr = threshold_at_fpr(1 - pv[:, 0], yv, 0.01)
        calibrated = True
        with open(model_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(clf, f)

    cfg.update({"window": args.window, "alarm_threshold": thr, "features": FEATURE_NAMES})
    if not args.final:
        prob, score = predict(Xv)
        res = {"model": args.model, "split": args.val_split, "config": cfg, "grid": grid,
               "metrics": evaluate(prob, yv, iv, mv, thr, score, calibrated),
               "minutes": (time.time() - t0) / 60}
        path = out_dir / f"{name}_val.json"
    else:
        res = {"model": args.model, "config": cfg, "metrics": {}}
        for sp in args.splits:
            Xs, ys, is_, ms = prepare(sp, args.window)
            prob, score = predict(Xs)
            res["metrics"][sp] = evaluate(prob, ys, is_, ms, thr, score, calibrated)
        res["minutes"] = (time.time() - t0) / 60
        path = out_dir / f"{name}_final.json"
    json.dump(res, open(path, "w"), indent=2)
    m = res["metrics"] if not args.final else res["metrics"][args.splits[0]]
    print(f"{name}: macro-F1 {m['macro_f1']:.4f} | recall lỗi @ báo nhầm<=1%: {m['recall_fault_at_thr']:.4f} "
          f"(trôi {m['recall_drift_at_thr']:.3f}, giả {m['recall_spoof_at_thr']:.3f}) | "
          f"trễ trung vị {m['median_delay_steps']} bước | bỏ sót {m['missed_faulty_rovers']:.3f} | "
          f"ECE {m['ece_fault']:.4f} | {res['minutes']:.1f} phút")
    print(f"Đã lưu {path}")


if __name__ == "__main__":
    main()
