"""E3 + E4 + E5: đánh giá cuối trên test và các tập ood_*. CHẠY MỘT LẦN, sau khi mọi việc dò/train đã xong.

Mọi ngưỡng báo động đều lấy từ val (đã lưu trong checkpoint, hoặc chọn lại trên val với baseline),
không có gì được chọn trên test.

Một lần chạy, cho mỗi tập:
  E3 / E5 : mọi thước đo (macro-F1, recall @ báo nhầm 1% trên val, trễ, bỏ sót, ECE) cho 6 mô hình,
            trung bình ± độ lệch chuẩn qua các seed báo cáo 10–29 (baseline tất định: 1 lần).
  E4      : recall tách theo loại lỗi × thời gian kể từ lúc lỗi bắt đầu, lúc không có người quan sát,
            và tỉ lệ báo nhầm ở rover trung thực đang đo với rover lỗi ("đổ oan hàng xóm").
  Kiểm định: khoảng tin cậy 95% bootstrap HAI TẦNG (lấy mẫu lại seed và episode, 2000 lần) cho chênh lệch
            GRU+GAT − GRU, GRU+GAT − GAT, GRU+GAT − logistic.

Checkpoint được tìm trong mọi results/runs_*/ckpt (cả Pro lẫn Free), nên một tài khoản chạy được hết.
Ghi: results/runs_<ACC>/E3_final_<split>.json

Chạy:
  python src/final_eval.py --splits test ood_drift8 ood_noise2 ood_two_faulty ood_n8
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as BL  # noqa: E402
from metrics import evaluate, threshold_at_fpr  # noqa: E402
from models import MODELS, SeqAE  # noqa: E402
from train_nn import DEV, ae_score, ae_windows, flat, load_split, predict, summarize  # noqa: E402

W = 30
SEEDS = list(range(10, 30))          # có checkpoint seed nào thì dùng seed đó (5 hoặc 20)
NN = ["ae", "gat", "gru", "grugat"]
ORDER = ["threshold", "logistic", "ae", "gat", "gru", "grugat"]
COMPARE = [("grugat", "gru"), ("grugat", "gat"), ("grugat", "logistic")]


def find_ckpt(model, seed):
    hits = sorted(glob.glob(f"results/runs_*/ckpt/E2_{model}_s{seed}.pt"))
    return hits[0] if hits else None


# dự đoán của từng mô hình
def baseline_predictors(w):
    """Dựng lại hai baseline đúng như lúc dò (tất định): ngưỡng chọn trên val, logistic fit trên train."""
    Xv, yv, _, _ = BL.prepare("val", w)
    cfg = BL.tune_threshold(Xv, yv)
    thr_pred = lambda X: (BL.onehot(BL.threshold_predict(X[:, 0], cfg["t1"], cfg["t2"])), X[:, 0], cfg["t1"], False)
    Xtr, ytr, _, _ = BL.prepare("train", w, 5)
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = BL.fit_logistic(Xtr, ytr, C)
        f = BL.f1_score(yv, clf.predict(Xv), average="macro", labels=[0, 1, 2], zero_division=0)
        if best is None or f > best[0]:
            best = (f, C, clf)
    clf = best[2]
    t_log = threshold_at_fpr(1 - clf.predict_proba(Xv)[:, 0], yv, 0.01)

    def log_pred(X):
        p = clf.predict_proba(X).astype(np.float32)
        return p, None, t_log, True
    print(f"  baseline: t1={cfg['t1']:.3f} t2={cfg['t2']:.3f} | logistic C={best[1]}", flush=True)
    return {"threshold": thr_pred, "logistic": log_pred}


def run_models(split, preds_bl):
    """Trả về dict (model, seed) -> (prob, score, thr, calibrated), cùng y, idx, D."""
    D = load_split(split)
    d_raw, _ = BL.load(split)
    X = BL.window_features(d_raw, W)
    Xf, y, idx = BL.flatten(X, d_raw["label"].astype(int), W)
    out = {}
    for name, fn in preds_bl.items():
        out[(name, 0)] = fn(Xf)
    for m in NN:
        for s in SEEDS:
            path = find_ckpt(m, s)
            if path is None:
                print(f"  THIẾU checkpoint {m} seed {s} — bỏ qua", flush=True)
                continue
            ck = torch.load(path, map_location=DEV, weights_only=False)
            if m == "ae":
                net = SeqAE(hidden=ck["cfg"]["hidden"], window=W).to(DEV)
                net.load_state_dict(ck["state"])
                Xa, ya = ae_windows(D, W)
                sc = ae_score(net, Xa, ck["extra"]["mu"], ck["extra"]["sd"])
                c = (sc > ck["extra"]["t1"]).astype(int) + (sc > ck["extra"]["t2"]).astype(int)
                assert np.array_equal(ya, y)
                out[(m, s)] = (np.eye(3, dtype=np.float32)[c], sc, ck["extra"]["t1"], False)
            else:
                net = MODELS[m](hidden=ck["cfg"]["hidden"]).to(DEV)
                net.load_state_dict(ck["state"])
                p, yy, ii = flat(predict(net, D), D["label"], W)
                assert np.array_equal(yy, y) and np.array_equal(ii, idx)
                out[(m, s)] = (p, None, ck["thr"], True)
    return out, y, idx, D


# nhóm phân tích (E4)
def make_groups(y, idx, D):
    meta = D["meta"]
    e_, t_, n_ = idx[:, 0], idx[:, 1], idx[:, 2]
    ftype = np.array([m["fault_type"] for m in meta])[e_]
    tu = np.array([m["t_u"] for m in meta])[e_]
    dt = t_ - tu
    obs = D["valid"].sum(axis=2)[e_, t_, n_]                    # số người quan sát rover n tại bước t
    honest_nb = np.zeros(len(y), bool)                           # trung thực và đang đo với >= 1 rover lỗi
    for ei, mm in enumerate(meta):
        if not mm["faulty"]:
            continue
        sel = np.where((e_ == ei) & (y == 0))[0]
        if len(sel) == 0:
            continue
        v = D["valid"][ei]                                       # (T,N,N), [i, k]
        hit = np.zeros(len(sel), bool)
        for f in mm["faulty"]:
            hit |= v[t_[sel], f, n_[sel]]
        honest_nb[sel] = hit
    G = {}
    for k in ["drift", "spoof", "spoof_one"]:
        base = (y > 0) & (ftype == k)
        if not base.any():
            continue
        G[f"{k} | tất cả"] = base
        for lo, hi in [(0, 30), (30, 100), (100, 400)]:
            G[f"{k} | {lo}-{hi} bước sau lỗi"] = base & (dt >= lo) & (dt < hi)
        G[f"{k} | 0 người quan sát"] = base & (obs == 0)
    G["trung thực | đang đo với rover lỗi"] = honest_nb
    G["trung thực | còn lại"] = (y == 0) & ~honest_nb
    return G


# bootstrap theo episode
def episode_stats(prob, score, thr, y, idx, E, groups):
    """Thống kê cộng được theo episode: ma trận nhầm lẫn 3x3, và (số báo động, số mẫu) cho từng nhóm."""
    pred = prob.argmax(1)
    alarm = ((1 - prob[:, 0]) if score is None else score) >= thr
    e_ = idx[:, 0]
    conf = np.zeros((E, 3, 3))
    np.add.at(conf, (e_, y, pred), 1)
    gs = {}
    for g, sel in groups.items():
        a = np.bincount(e_[sel], weights=alarm[sel], minlength=E)
        n = np.bincount(e_[sel], minlength=E).astype(float)
        gs[g] = (a, n)
    return conf, gs


def macro_f1_from_conf(c):
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    prec = tp / np.maximum(c.sum(-2), 1)
    rec = tp / np.maximum(c.sum(-1), 1)
    f = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return f.mean(-1)


def bootstrap(stats, E, B=2000, seed=0):
    """Bootstrap HAI TẦNG: mỗi lần lấy mẫu lại cả episode (dao động dữ liệu) lẫn seed huấn luyện
    (dao động do train lại). Episode được lấy chung cho mọi mô hình (ghép cặp theo episode);
    seed lấy độc lập cho từng mô hình (các mô hình train độc lập).
    stats[model] = list theo seed của (conf, gs). Trả về mẫu bootstrap (B,) của từng thước đo."""
    rng = np.random.default_rng(seed)
    W_ = np.stack([np.bincount(rng.integers(0, E, E), minlength=E) for _ in range(B)]).astype(float)  # (B,E)
    res = {}
    for m, lst in stats.items():
        n_s = len(lst)
        pick = rng.integers(0, n_s, (B, n_s))                          # seed lấy lại cho từng lần bootstrap
        cols = np.arange(B)[:, None]
        mf = np.stack([macro_f1_from_conf(np.einsum("be,eij->bij", W_, conf)) for conf, _ in lst])  # (n_s,B)
        out = {"macro_f1": mf[pick, cols].mean(1)}
        for g in lst[0][1]:
            M = np.stack([(W_ @ gs[g][0]) / np.maximum(W_ @ gs[g][1], 1) for _, gs in lst])      # (n_s,B)
            out[g] = M[pick, cols].mean(1)
        res[m] = out
    return res


def ci(x):
    return [float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# chương trình chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    acc = os.environ.get("ACC", "local")
    out_dir = Path(f"results/runs_{acc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Thiết bị: {DEV}", flush=True)
    t0 = time.time()
    preds_bl = baseline_predictors(W)

    for sp in args.splits:
        ts = time.time()
        print(f"\n=== {sp}", flush=True)
        out, y, idx, D = run_models(sp, preds_bl)
        E = len(D["meta"])
        groups = make_groups(y, idx, D)
        groups_boot = {"fault": y > 0, "honest": y == 0,
                       "trung thực | đang đo với rover lỗi": groups["trung thực | đang đo với rover lỗi"]}
        res = {"split": sp, "n_episodes": E, "n_samples": int(len(y)), "models": {}, "groups": {},
               "bootstrap_diff": {}, "group_sizes": {g: int(s.sum()) for g, s in groups.items()}}
        stats = {}
        for m in ORDER:
            keys = [k for k in out if k[0] == m]
            if not keys:
                continue
            per_seed, grp = [], {g: [] for g in groups}
            for k in keys:
                prob, score, thr, cal = out[k]
                per_seed.append(evaluate(prob, y, idx, D["meta"], thr, score, cal))
                alarm = ((1 - prob[:, 0]) if score is None else score) >= thr
                for g, sel in groups.items():
                    grp[g].append(float(alarm[sel].mean()) if sel.any() else float("nan"))
                stats.setdefault(m, []).append(episode_stats(prob, score, thr, y, idx, E, groups_boot))
            res["models"][m] = {"n_seeds": len(keys), "summary": summarize(per_seed), "per_seed": per_seed}
            res["groups"][m] = {g: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                                for g, v in grp.items()}
        bs = bootstrap(stats, E, args.boot)
        for a, b in COMPARE:
            if a in bs and b in bs:
                res["bootstrap_diff"][f"{a} - {b}"] = {
                    "macro_f1": ci(bs[a]["macro_f1"] - bs[b]["macro_f1"]),
                    "recall_fault_at_thr": ci(bs[a]["fault"] - bs[b]["fault"]),
                    "fpr_at_thr": ci(bs[a]["honest"] - bs[b]["honest"]),
                    "fpr_honest_neighbor": ci(bs[a]["trung thực | đang đo với rover lỗi"]
                                              - bs[b]["trung thực | đang đo với rover lỗi"]),
                }
        res["minutes"] = (time.time() - ts) / 60
        path = out_dir / f"E3_final_{sp}.json"
        json.dump(res, open(path, "w"), indent=1)

        # in tóm tắt
        print(f"{'mô hình':10s} {'macro-F1':>15s} {'recall@1%':>15s} {'FPR':>7s} {'trễ':>6s} {'bỏ sót':>7s} {'ECE':>6s}")
        for m, r in res["models"].items():
            s = r["summary"]
            print(f"{m:10s} {s['macro_f1']['mean']:.3f}±{s['macro_f1']['std']:.3f}   "
                  f"{s['recall_fault_at_thr']['mean']:.3f}±{s['recall_fault_at_thr']['std']:.3f}   "
                  f"{s['fpr_at_thr']['mean']:.3f} {s['median_delay_steps']['mean']:6.1f} "
                  f"{s['missed_faulty_rovers']['mean']:7.3f} {s['ece_fault']['mean']:6.3f}")
        show = [m for m in ["logistic", "gat", "gru", "grugat"] if m in res["groups"]]
        print(f"\n{'nhóm (tỉ lệ báo động)':40s}" + "".join(f"{m:>10s}" for m in show))
        for g in groups:
            print(f"{g:40s}" + "".join(f"{res['groups'][m][g]['mean']:10.3f}" for m in show))
        print("\nChênh lệch, trung bình [KTC 95% bootstrap hai tầng: seed × episode]:")
        for k, v in res["bootstrap_diff"].items():
            print(f"  {k:20s} " + " | ".join(f"{mm} {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]" for mm, x in v.items()))
        print(f"-> {path}  ({res['minutes']:.1f} phút)", flush=True)
    print(f"\nTổng: {(time.time()-t0)/60:.1f} phút")


if __name__ == "__main__":
    main()
