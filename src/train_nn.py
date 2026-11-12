"""E1 (autoencoder) và E2 (GRU, GAT, GRU + GAT): huấn luyện, dò siêu tham số, đánh giá.

Ba chế độ, chạy theo thứ tự:
  --mode tune    : thử 4 cấu hình (hidden ∈ {32, 64} × lr ∈ {1e-3, 3e-3}) với seed dò,
                   early stopping theo macro-F1 trên val, chọn cấu hình tốt nhất.
                   -> results/runs_<ACC>/E2_<model>_tune.json
  --mode report  : train lại cấu hình đã chọn với 5 seed báo cáo (configs/seeds.yaml),
                   lưu checkpoint và kết quả val của từng seed.
                   -> results/runs_<ACC>/E2_<model>_report.json, results/runs_<ACC>/ckpt/E2_<model>_s<seed>.pt
  --mode final   : CHỈ CHẠY MỘT LẦN vào ngày đánh giá. Nạp checkpoint 5 seed, chọn ngưỡng báo động
                   trên val, đánh giá trên test và các tập ood_*.
                   -> results/runs_<ACC>/E3_<model>_final.json

Ví dụ:
  python src/train_nn.py --model grugat --mode tune
  python src/train_nn.py --model grugat --mode report
  python src/train_nn.py --model grugat --mode final --splits test ood_drift8 ood_noise2 ood_two_faulty ood_n8
  python src/train_nn.py --model ae --mode tune          # autoencoder (baseline 3)
Chạy thử nhanh: thêm --max_train 100 --max_val 50 --epochs 2
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, threshold_at_fpr  # noqa: E402
from models import MODELS, SeqAE  # noqa: E402
from tensors import build  # noqa: E402

GRID = {"hidden": [32, 64], "lr": [1e-3, 3e-3]}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ dữ liệu
def transform(D):
    """Nén thang đo: phần dư có thể từ vài cm tới hàng chục mét (khai giả)."""
    e = D["edge"]
    e = np.stack([np.sign(e[..., 0]) * np.log1p(e[..., 0] * np.sign(e[..., 0])),
                  np.log1p(e[..., 1]), e[..., 2]], -1).astype(np.float32)
    n = D["node"].copy()
    n[..., :3] = np.log1p(n[..., :3])
    n[..., 3] = np.sign(n[..., 3]) * np.log1p(np.abs(n[..., 3]))
    n[..., 4] = n[..., 4] / 4.0
    n[..., 5] = n[..., 5] / 0.3
    return dict(D, edge=e, node=n.astype(np.float32))


def load_split(split, max_ep=None):
    D = transform(build(split))
    if max_ep:
        D = {k: (v[:max_ep] if k != "meta" else v[:max_ep]) for k, v in D.items()}
    return D


def to_batch(D, ids):
    return {k: torch.from_numpy(D[k][ids]).to(DEV) for k in ("edge", "node", "valid", "recent", "label")}


def flat(prob_etn, label_etn, w):
    """Lấy các bước t >= w-1 (giống baseline), trả (prob, y, idx) theo thứ tự (e, t, n)."""
    E, T, N, _ = prob_etn.shape
    ts = np.arange(w - 1, T)
    ee, tt, nn_ = np.meshgrid(np.arange(E), ts, np.arange(N), indexing="ij")
    return (prob_etn[:, ts].reshape(-1, 3), label_etn[:, ts].reshape(-1),
            np.stack([ee, tt, nn_], -1).reshape(-1, 3))


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------ mô hình có giám sát
@torch.no_grad()
def predict(model, D, bs=16):
    model.eval()
    out = []
    for s in range(0, len(D["label"]), bs):
        ids = np.arange(s, min(s + bs, len(D["label"])))
        out.append(torch.softmax(model(to_batch(D, ids)), -1).float().cpu().numpy())
    return np.concatenate(out)


def class_weights(D, w):
    y = D["label"][:, w - 1:].ravel()
    cnt = np.bincount(y, minlength=3).astype(np.float64)
    wts = cnt.sum() / (3 * np.maximum(cnt, 1))
    return torch.tensor(wts, dtype=torch.float32, device=DEV)


def train_supervised(model_name, hidden, lr, seed, Dtr, Dv, args):
    set_seed(seed)
    model = MODELS[model_name](hidden=hidden).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(Dtr, args.window))
    w0 = args.window - 1
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(Dtr["label"]))
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = to_batch(Dtr, np.sort(perm[s:s + args.batch]))
            logits = model(b)[:, w0:]
            loss = loss_fn(logits.reshape(-1, 3), b["label"][:, w0:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b["label"])
        pv = predict(model, Dv)
        p, y, _ = flat(pv, Dv["label"], args.window)
        f = f1_score(y, p.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)
        # loss trên val để vẽ đường cong train/val (phân tích overfitting)
        with torch.no_grad():
            lv = nn.functional.nll_loss(torch.log(torch.from_numpy(p).clamp_min(1e-8)),
                                        torch.from_numpy(y), weight=loss_fn.weight.cpu()).item()
        history.append({"epoch": ep, "train_loss": tot / len(perm), "val_loss": lv, "val_macro_f1": float(f)})
        print(f"    ep {ep:2d}  train_loss {tot/len(perm):.4f}  val_loss {lv:.4f}  val_macroF1 {f:.4f}  "
              f"({(time.time()-t0)/60:.1f} phút)", flush=True)
        if f > best:
            best, best_ep, bad = f, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep, "val_macro_f1": float(best), "history": history,
                   "minutes": (time.time() - t0) / 60,
                   "params": int(sum(p.numel() for p in model.parameters()))}


# ------------------------------------------------------------------ autoencoder
def ae_windows(D, w, stride=1):
    from numpy.lib.stride_tricks import sliding_window_view
    x = sliding_window_view(D["node"], w, axis=1)[:, ::stride]          # (E,T',N,6,W)
    x = np.ascontiguousarray(x.transpose(0, 1, 2, 4, 3))                # (E,T',N,W,6)
    E, Tp, N = x.shape[:3]
    y = D["label"][:, w - 1:][:, ::stride]
    return x.reshape(E * Tp * N, -1), y.reshape(-1)


@torch.no_grad()
def ae_score(model, X, mu, sd, bs=8192):
    model.eval()
    out = []
    for s in range(0, len(X), bs):
        x = torch.from_numpy((X[s:s + bs] - mu) / sd).to(DEV)
        out.append(((model(x) - x) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out)


def two_thresholds(s, y):
    t1 = threshold_at_fpr(s, y, 0.01)
    cands = np.quantile(s[s > t1], np.linspace(0.02, 0.98, 49)) if (s > t1).any() else [t1]
    best = max((f1_score(y, (s > t1).astype(int) + (s > t2).astype(int), average="macro",
                         labels=[0, 1, 2], zero_division=0), float(t2)) for t2 in cands)
    return t1, best[1], best[0]


def train_ae(hidden, lr, seed, Dtr, Dv, args):
    set_seed(seed)
    Xtr, ytr = ae_windows(Dtr, args.window, stride=5)
    Xtr = Xtr[ytr == 0]                                   # chỉ học dữ liệu trung thực
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xv, yv = ae_windows(Dv, args.window)
    model = SeqAE(hidden=hidden, window=args.window).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    best, best_state, best_ep, bad, t0, history = -1.0, None, 0, 0, time.time(), []
    Xt = torch.from_numpy((Xtr - mu) / sd)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(Xt))
        tot = 0.0
        for s in range(0, len(perm), 1024):
            x = Xt[perm[s:s + 1024]].to(DEV)
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sv = ae_score(model, Xv, mu, sd)
        t1, t2, f = two_thresholds(sv, yv)
        history.append({"epoch": ep, "train_loss": tot / len(perm), "val_macro_f1": float(f)})
        print(f"    ep {ep:2d}  train_recon {tot/len(perm):.4f}  val_macroF1 {f:.4f}  "
              f"({(time.time()-t0)/60:.1f} phút)", flush=True)
        if f > best:
            best, best_ep, bad = f, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            thr = (t1, t2)
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    extra = {"mu": mu, "sd": sd, "t1": thr[0], "t2": thr[1]}
    return model, extra, {"best_epoch": best_ep, "val_macro_f1": float(best), "history": history,
                          "minutes": (time.time() - t0) / 60,
                          "params": int(sum(p.numel() for p in model.parameters()))}


def ae_eval(model, extra, D, w):
    X, y = ae_windows(D, w)
    s = ae_score(model, X, extra["mu"], extra["sd"])
    c = (s > extra["t1"]).astype(int) + (s > extra["t2"]).astype(int)
    prob = np.eye(3, dtype=np.float32)[c]
    E, T, N = D["label"].shape
    ts = np.arange(w - 1, T)
    ee, tt, nn_ = np.meshgrid(np.arange(E), ts, np.arange(N), indexing="ij")
    idx = np.stack([ee, tt, nn_], -1).reshape(-1, 3)
    return evaluate(prob, y, idx, D["meta"], extra["t1"], score=s, calibrated=False)


# ------------------------------------------------------------------ chương trình chính
def summarize(per_seed):
    keys = ["macro_f1", "recall_fault_at_thr", "recall_drift_at_thr", "recall_spoof_at_thr",
            "median_delay_steps", "missed_faulty_rovers", "ece_fault", "fpr_at_thr"]
    out = {}
    for k in keys:
        v = np.array([m[k] for m in per_seed], float)
        v = v[np.isfinite(v)]
        if len(v) == 0:                                   # ví dụ ECE của autoencoder (không ra xác suất)
            out[k] = {"mean": float("nan"), "std": float("nan")}
            continue
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gru", "gat", "grugat", "ae"], required=True)
    ap.add_argument("--mode", choices=["tune", "report", "final"], required=True)
    ap.add_argument("--window", type=int, default=30, help="bước đầu tiên được đánh giá = window-1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--tag", default="", help="hậu tố tên file, ví dụ _gauss")
    ap.add_argument("--hidden", type=int, default=None, help="ghi đè cấu hình (dùng cho E6)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_train", type=int, default=None, help="chỉ để chạy thử")
    ap.add_argument("--max_val", type=int, default=None, help="chỉ để chạy thử")
    args = ap.parse_args()

    acc = os.environ.get("ACC", "local")
    out_dir = Path(f"results/runs_{acc}")
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    seeds = yaml.safe_load(open("configs/seeds.yaml"))
    name = f"E2_{args.model}{args.tag}"
    print(f"Thiết bị: {DEV}", flush=True)

    t_load = time.time()
    Dv = load_split(args.val_split, args.max_val)
    Dtr = load_split(args.train_split, args.max_train) if args.mode != "final" else None
    print(f"Nạp dữ liệu: {(time.time()-t_load)/60:.1f} phút", flush=True)

    if args.mode == "tune":
        res = {"model": args.model, "grid": [], "seed": seeds["tune_seeds"][0]}
        for hidden, lr in itertools.product(GRID["hidden"], GRID["lr"]):
            print(f"  cấu hình hidden={hidden} lr={lr}", flush=True)
            if args.model == "ae":
                _, _, info = train_ae(hidden, lr, res["seed"], Dtr, Dv, args)
            else:
                _, info = train_supervised(args.model, hidden, lr, res["seed"], Dtr, Dv, args)
            res["grid"].append({"hidden": hidden, "lr": lr, **info})
        best = max(res["grid"], key=lambda g: g["val_macro_f1"])
        res["best"] = {"hidden": best["hidden"], "lr": best["lr"], "val_macro_f1": best["val_macro_f1"]}
        path = out_dir / f"{name}_tune.json"
        json.dump(res, open(path, "w"), indent=1)
        print(f"Cấu hình tốt nhất: {res['best']}  -> {path}")
        return

    tune_path = out_dir / f"E2_{args.model}_tune.json"
    if args.hidden and args.lr:
        cfg = {"hidden": args.hidden, "lr": args.lr}
    else:
        cfg = json.load(open(tune_path))["best"]
    print(f"Cấu hình: hidden={cfg['hidden']} lr={cfg['lr']}", flush=True)

    if args.mode == "report":
        res = {"model": args.model, "config": cfg, "seeds": {}}
        per_seed = []
        for s in seeds["report_seeds"]:
            print(f"  seed {s}", flush=True)
            ck = out_dir / "ckpt" / f"{name}_s{s}.pt"
            if args.model == "ae":
                model, extra, info = train_ae(cfg["hidden"], cfg["lr"], s, Dtr, Dv, args)
                m = ae_eval(model, extra, Dv, args.window)
                torch.save({"state": model.state_dict(), "cfg": cfg, "extra": extra}, ck)
            else:
                model, info = train_supervised(args.model, cfg["hidden"], cfg["lr"], s, Dtr, Dv, args)
                pv = predict(model, Dv)
                p, y, idx = flat(pv, Dv["label"], args.window)
                thr = threshold_at_fpr(1 - p[:, 0], y, 0.01)
                t_inf = time.time()
                predict(model, {k: (v[:50] if k != "meta" else v) for k, v in Dv.items()})
                info["infer_ms_per_episode"] = (time.time() - t_inf) / 50 * 1000
                m = evaluate(p, y, idx, Dv["meta"], thr)
                torch.save({"state": model.state_dict(), "cfg": cfg, "thr": thr}, ck)
            res["seeds"][str(s)] = {"val": m, **info}
            per_seed.append(m)
            print(f"  seed {s}: val macro-F1 {m['macro_f1']:.4f} | recall lỗi @1%: {m['recall_fault_at_thr']:.4f}",
                  flush=True)
        res["summary_val"] = summarize(per_seed)
        path = out_dir / f"{name}_report.json"
        json.dump(res, open(path, "w"), indent=1)
        sv = res["summary_val"]
        print(f"{name} (val, 5 seed): macro-F1 {sv['macro_f1']['mean']:.4f} ± {sv['macro_f1']['std']:.4f} | "
              f"recall lỗi @1% {sv['recall_fault_at_thr']['mean']:.4f} ± {sv['recall_fault_at_thr']['std']:.4f}  -> {path}")
        return

    # ---------------- final: đánh giá cuối, chạy một lần
    res = {"model": args.model, "config": cfg, "splits": {}}
    tests = {sp: load_split(sp) for sp in args.splits}
    for s in seeds["report_seeds"]:
        ck = torch.load(out_dir / "ckpt" / f"{name}_s{s}.pt", map_location=DEV, weights_only=False)
        if args.model == "ae":
            model = SeqAE(hidden=cfg["hidden"], window=args.window).to(DEV)
            model.load_state_dict(ck["state"])
            for sp, D in tests.items():
                res["splits"].setdefault(sp, []).append(ae_eval(model, ck["extra"], D, args.window))
        else:
            model = MODELS[args.model](hidden=cfg["hidden"]).to(DEV)
            model.load_state_dict(ck["state"])
            thr = ck["thr"]                                   # ngưỡng đã chọn trên val
            for sp, D in tests.items():
                p, y, idx = flat(predict(model, D), D["label"], args.window)
                res["splits"].setdefault(sp, []).append(evaluate(p, y, idx, D["meta"], thr))
    res["summary"] = {sp: summarize(v) for sp, v in res["splits"].items()}
    path = out_dir / f"E3_{args.model}{args.tag}_final.json"
    json.dump(res, open(path, "w"), indent=1)
    for sp, sm in res["summary"].items():
        print(f"{sp}: macro-F1 {sm['macro_f1']['mean']:.4f} ± {sm['macro_f1']['std']:.4f} | "
              f"recall lỗi @1% {sm['recall_fault_at_thr']['mean']:.4f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
