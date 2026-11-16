"""E7: so sánh CÔNG BẰNG giữa các mô hình khi điều kiện thay đổi (thêm ngày 14/11 sau khi xem E5).

Vấn đề: ở E5 mọi ngưỡng lấy từ val chuẩn (1 rover lỗi, nhiễu thường). Khi điều kiện đổi, mỗi mô hình
rơi vào một mức báo nhầm khác nhau (ví dụ trên ood_two_faulty: GRU 5,6%, GRU+GAT 1,4%), nên so recall
trực tiếp là không công bằng. Ba cách so công bằng, từ chặt tới lỏng:

  A. Không cần ngưỡng: AUROC và AUPRC của điểm "có lỗi" (xếp hạng mẫu lỗi trên mẫu trung thực).
  B. Ngưỡng hiệu chỉnh lại trên một tập val CÙNG điều kiện, seed riêng (val_two_faulty, val_noise2):
     mỗi mô hình đặt ngưỡng sao cho báo nhầm 1% trên tập val đó, rồi đo trên tập đánh giá.
     Đây là quy trình hợp lệ: không có gì được chọn trên tập đánh giá.
  C. Ngưỡng "oracle" chọn ngay trên tập đánh giá cho báo nhầm đúng 1%. CHỈ để tham khảo (lạc quan).

Kèm theo: chẩn đoán trên tập 2 rover lỗi — recall khi rover lỗi đang đo với rover lỗi kia hay không.
Đánh giá cả các biến thể (đuôi _mf: train thêm dữ liệu 2 rover lỗi; _gauss: train bằng nhiễu Gaussian)
nếu có checkpoint.

Ghi: results/runs_<ACC>/E7_fair_<split>.json
Chạy:
  python src/fair_eval.py --pairs test:val ood_two_faulty:val_two_faulty ood_two_faulty_b:val_two_faulty \
      ood_noise2:val_noise2 ood_n8:- ood_drift8:-
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as BL  # noqa: E402
from final_eval import W, SEEDS, baseline_predictors, bootstrap, ci, episode_stats, find_ckpt  # noqa: E402
from metrics import threshold_at_fpr  # noqa: E402
from models import MODELS, SeqAE  # noqa: E402
from train_nn import DEV, ae_score, ae_windows, flat, load_split, predict  # noqa: E402

BASE = ["ae", "gat", "gru", "grugat"]
TAGS = ["", "_mf", "_more", "_gauss"]


def available_models():
    out = []
    for b in BASE:
        for t in TAGS:
            seeds = [s for s in SEEDS if find_ckpt(b + t, s)]
            if seeds:
                out.append((b + t, b, seeds))
    return out


def scores_on(split, preds_bl, models):
    """dict tên -> list theo seed của (điểm lỗi, ngưỡng gốc từ val chuẩn). Cùng y, idx, D."""
    D = load_split(split)
    d_raw, _ = BL.load(split)
    Xf, y, idx = BL.flatten(BL.window_features(d_raw, W), d_raw["label"].astype(int), W)
    out = {}
    for name, fn in preds_bl.items():
        prob, sc, thr, _ = fn(Xf)
        out[name] = [((1 - prob[:, 0]) if sc is None else sc, thr)]
    for name, base, seeds in models:
        for s in seeds:
            ck = torch.load(find_ckpt(name, s), map_location=DEV, weights_only=False)
            if base == "ae":
                net = SeqAE(hidden=ck["cfg"]["hidden"], window=W).to(DEV)
                net.load_state_dict(ck["state"])
                Xa, _ = ae_windows(D, W)
                out.setdefault(name, []).append((ae_score(net, Xa, ck["extra"]["mu"], ck["extra"]["sd"]),
                                                 ck["extra"]["t1"]))
            else:
                net = MODELS[base](hidden=ck["cfg"]["hidden"]).to(DEV)
                net.load_state_dict(ck["state"])
                p, yy, ii = flat(predict(net, D), D["label"], W)
                assert np.array_equal(yy, y) and np.array_equal(ii, idx)
                out.setdefault(name, []).append((1 - p[:, 0], ck["thr"]))
    return out, y, idx, D


def fault_groups(y, idx, D):
    """Nhóm chẩn đoán: mẫu lỗi đang/không đo với rover lỗi khác; hàng xóm trung thực của rover lỗi."""
    meta = D["meta"]
    e_, t_, n_ = idx[:, 0], idx[:, 1], idx[:, 2]
    nb_fault = np.zeros(len(y), bool)      # đang đo với ít nhất một rover lỗi KHÁC chính nó
    for ei, mm in enumerate(meta):
        if not mm["faulty"]:
            continue
        sel = np.where(e_ == ei)[0]
        v = D["valid"][ei]
        for f in mm["faulty"]:
            hit = v[t_[sel], f, n_[sel]] & (n_[sel] != f)
            nb_fault[sel[hit]] = True
    G = {"lỗi | tất cả": y > 0,
         "lỗi | đang đo với rover lỗi khác": (y > 0) & nb_fault,
         "lỗi | không đo với rover lỗi khác": (y > 0) & ~nb_fault,
         "trung thực | đang đo với rover lỗi": (y == 0) & nb_fault}
    return {g: s for g, s in G.items() if s.any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="tập_đánh_giá:tập_val_hiệu_chỉnh (hoặc :-)")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    acc = os.environ.get("ACC", "local")
    out_dir = Path(f"results/runs_{acc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    models = available_models()
    print(f"Thiết bị: {DEV} | mô hình: " + ", ".join(f"{n}({len(s)} seed)" for n, _, s in models), flush=True)
    preds_bl = baseline_predictors(W)
    calib_cache = {}

    for pair in args.pairs:
        sp, cal = pair.split(":")
        print(f"\n=== {sp}   (hiệu chỉnh ngưỡng trên: {cal})", flush=True)
        S, y, idx, D = scores_on(sp, preds_bl, models)
        if cal != "-" and cal not in calib_cache:
            Sc, yc, _, _ = scores_on(cal, preds_bl, models)
            calib_cache[cal] = {m: [threshold_at_fpr(sc, yc, 0.01) for sc, _ in v] for m, v in Sc.items()}
        thr_cal = calib_cache.get(cal)
        G = fault_groups(y, idx, D)
        E = len(D["meta"])
        res = {"split": sp, "calib": cal, "models": {}, "bootstrap_diff": {}}
        stats = {}
        for m, lst in S.items():
            rows = []
            for k, (sc, thr0) in enumerate(lst):
                r = {"auroc": float(roc_auc_score(y > 0, sc)), "auprc": float(average_precision_score(y > 0, sc))}
                thrs = {"val_chuan": thr0, "oracle": threshold_at_fpr(sc, y, 0.01)}
                if thr_cal:
                    thrs["val_cung_dk"] = thr_cal[m][k]
                for tn, th in thrs.items():
                    a = sc >= th
                    r[tn] = {"thr": float(th), "fpr": float(a[y == 0].mean()), "recall": float(a[y > 0].mean()),
                             "groups": {g: float(a[s].mean()) for g, s in G.items()}}
                rows.append(r)
                if thr_cal:   # bootstrap ở ngưỡng hiệu chỉnh cùng điều kiện
                    prob = np.stack([1 - sc, sc / 2, sc / 2], 1)       # chỉ dùng cột 0 cho báo động
                    stats.setdefault(m, []).append(episode_stats(prob, None, thr_cal[m][k], y, idx, E,
                                                                 {"fault": y > 0, "honest": y == 0}))
            agg = lambda f: (float(np.mean([f(r) for r in rows])),
                             float(np.std([f(r) for r in rows], ddof=1)) if len(rows) > 1 else 0.0)
            summ = {"auroc": agg(lambda r: r["auroc"]), "auprc": agg(lambda r: r["auprc"])}
            for tn in rows[0]:
                if isinstance(rows[0][tn], dict):
                    summ[tn] = {"fpr": agg(lambda r: r[tn]["fpr"]), "recall": agg(lambda r: r[tn]["recall"]),
                                "groups": {g: agg(lambda r: r[tn]["groups"][g]) for g in G}}
            res["models"][m] = {"n_seeds": len(rows), "summary": summ, "per_seed": rows}
        if stats:
            bs = bootstrap(stats, E, args.boot)
            for a, b in [("grugat", "gru"), ("grugat_mf", "gru_mf"), ("grugat_mf", "grugat"), ("grugat_mf", "grugat_more"),
                         ("grugat_more", "grugat"), ("gru_mf", "gru")]:
                if a in bs and b in bs:
                    res["bootstrap_diff"][f"{a} - {b}"] = {"recall": ci(bs[a]["fault"] - bs[b]["fault"]),
                                                           "fpr": ci(bs[a]["honest"] - bs[b]["honest"])}
        path = out_dir / f"E7_fair_{sp}.json"
        json.dump(res, open(path, "w"), indent=1)

        # --- in bảng
        cols = ["val_chuan"] + (["val_cung_dk"] if thr_cal else []) + ["oracle"]
        hdr = {"val_chuan": "ngưỡng val chuẩn", "val_cung_dk": "ngưỡng val cùng đk", "oracle": "oracle (tham khảo)"}
        print(f"{'mô hình':12s} {'AUROC':>13s} {'AUPRC':>13s} " + " ".join(f"{hdr[c]:>24s}" for c in cols))
        print(f"{'':40s}" + " ".join(f"{'recall':>13s} {'FPR':>10s}" for _ in cols))
        for m, r in res["models"].items():
            s = r["summary"]
            line = f"{m:12s} {s['auroc'][0]:.4f}±{s['auroc'][1]:.4f} {s['auprc'][0]:.3f}±{s['auprc'][1]:.3f} "
            line += " ".join(f"{s[c]['recall'][0]:7.3f}±{s[c]['recall'][1]:.3f} {s[c]['fpr'][0]:10.3f}" for c in cols)
            print(line)
        c = "val_cung_dk" if thr_cal else "val_chuan"
        names = [m for m in ["logistic", "gru", "grugat", "gru_mf", "grugat_more", "grugat_mf", "grugat_gauss"] if m in res["models"]]
        print(f"\nTỉ lệ báo động theo nhóm, ở {hdr[c]}:")
        print(f"{'nhóm':38s}" + "".join(f"{m:>13s}" for m in names))
        for g in G:
            print(f"{g:38s}" + "".join(f"{res['models'][m]['summary'][c]['groups'][g][0]:13.3f}" for m in names)
                  + f"   (n={int(G[g].sum())})")
        if res["bootstrap_diff"]:
            print("\nChênh lệch ở ngưỡng val cùng điều kiện [KTC 95% bootstrap hai tầng: seed × episode]:")
            for k, v in res["bootstrap_diff"].items():
                print(f"  {k:22s} recall {v['recall'][0]:+.3f} [{v['recall'][1]:+.3f}, {v['recall'][2]:+.3f}] | "
                      f"FPR {v['fpr'][0]:+.3f} [{v['fpr'][1]:+.3f}, {v['fpr'][2]:+.3f}]")
        print(f"-> {path}", flush=True)
    print(f"\nTổng: {(time.time()-t0)/60:.1f} phút")


if __name__ == "__main__":
    main()
