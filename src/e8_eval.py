"""E8: đánh giá ghép cặp — cùng chính sách MARL, cùng 300 episode, đổi NGUỒN TIN CẬY (bộ phát hiện).

Giả thuyết viết TRƯỚC khi chạy (16/11):
  H1 (an toàn)    : diện tích "khai đã phủ nhưng thật ra chưa ai tới" (holes) trên episode có lỗi giảm dần
                    none > logistic > GRU >= GRU+GAT >= oracle.
  H2 (hiệu quả)   : đổ oan rover trung thực làm mất công phủ lại -> độ phủ thật giảm. Kiểm tra cơ chế bằng liều:
                    GRU+GAT + đổ oan nhân tạo cho hàng xóm trung thực của rover lỗi (0,7%, 2,1%, 10% mỗi mốc;
                    0,7% = phần GRU đổ oan nhiều hơn GRU+GAT ở E3) -> độ phủ giảm theo liều, holes không tăng.
  H3 (MARL)       : chính sách MAPPO phủ tốt hơn chính sách heuristic không học (cùng nguồn tin cậy).
Thước đo chính: holes (an toàn), độ phủ thật (hiệu quả), trên episode có lỗi; báo thêm episode không lỗi.
Hành động được LẤY MẪU từ chính sách (tốt hơn argmax ở thử nghiệm sơ bộ: 0,97 so với 0,94 độ phủ), với bộ sinh
số ngẫu nhiên cố định theo seed chính sách, giống nhau cho mọi điều kiện (common random numbers).
Số seed chính sách: 20 (10–29), khớp 20 seed của bộ phát hiện.

Bộ phát hiện chạy TRỰC TUYẾN: ở mỗi mốc quyết định t (từ bước 30), bộ phát hiện nhận dữ liệu bước 0..t-1
(vị trí tự khai + đo UWB do môi trường sinh) và báo động theo đúng ngưỡng đã chọn trên val ở E3 (FPR 1%/bước).
Chính sách seed s dùng bộ phát hiện seed s (GRU, GRU+GAT có 20 seed 10–29).

Chạy:  python src/e8_eval.py --seeds 10 11 12 13 14 --episodes 300
Ghi:   results/runs_<ACC>/e8/eval_s<seed>.json
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
from e8_env import N_DEC, CoverageEnv, heuristic_actions  # noqa: E402
from e8_train import ActorCritic, act  # noqa: E402
from features import window_features  # noqa: E402
from metrics import threshold_at_fpr  # noqa: E402
from models import MODELS  # noqa: E402
from tensors import build_arrays  # noqa: E402
from train_nn import DEV, predict, transform  # noqa: E402

EVAL_SEED0 = 10 ** 6
CONDITIONS = [  # (tên, chính sách, nguồn tin cậy, bộ phát hiện, liều đổ oan)
    ("none", "marl", "none", None, 0.0),
    ("oracle", "marl", "oracle", None, 0.0),
    ("logistic", "marl", "callback", "logistic", 0.0),
    ("gru", "marl", "callback", "gru", 0.0),
    ("grugat", "marl", "callback", "grugat", 0.0),
    ("grugat+blame0.7%", "marl", "callback", "grugat", 0.007),
    ("grugat+blame2.1%", "marl", "callback", "grugat", 0.021),
    ("grugat+blame10%", "marl", "callback", "grugat", 0.10),
    ("heur_oracle", "heuristic", "oracle", None, 0.0),
    ("heur_grugat", "heuristic", "callback", "grugat", 0.0),
]


def find(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def logistic_detector():
    """Dựng lại baseline logistic đúng như E1/E3 (fit trên train, C và ngưỡng chọn trên val)."""
    Xv, yv, _, _ = BL.prepare("val", 30)
    Xtr, ytr, _, _ = BL.prepare("train", 30, 5)
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = BL.fit_logistic(Xtr, ytr, C)
        f = BL.f1_score(yv, clf.predict(Xv), average="macro", labels=[0, 1, 2], zero_division=0)
        if best is None or f > best[0]:
            best = (f, clf)
    clf = best[1]
    thr = threshold_at_fpr(1 - clf.predict_proba(Xv)[:, 0], yv, 0.01)

    def score(data):
        X = window_features(data, 30)[:, -1]                    # (B,N,F) ở bước mới nhất
        B, N, F = X.shape
        return (1 - clf.predict_proba(X.reshape(-1, F))[:, 0]).reshape(B, N), thr
    return score


def nn_detector(name, seed):
    ck = torch.load(find(f"results/runs_*/ckpt/E2_{name}_s{seed}.pt"), map_location=DEV, weights_only=False)
    net = MODELS[name](hidden=ck["cfg"]["hidden"]).to(DEV)
    net.load_state_dict(ck["state"])
    net.eval()

    def score(data):
        D = transform(build_arrays(data))
        prob = predict(net, D)                                  # (B,t,N,3)
        return 1 - prob[:, -1, :, 0], ck["thr"]
    return score


def make_callback(score_fn, dose, ep_seeds):
    rngs = [np.random.default_rng([999, int(s)]) for s in ep_seeds]

    def cb(env):
        sc, thr = score_fn(env.data_prefix())
        fl = sc >= thr
        if dose > 0:                                            # đổ oan nhân tạo cho hàng xóm trung thực
            t = env.t - 1
            for b in np.where(env.faulty >= 0)[0]:
                f = env.faulty[b]
                if env.label[b, t, f] == 0:
                    continue
                for k in range(env.N):
                    if k != f and env.label[b, t, k] == 0 and np.isfinite(env.r[b, t, k, f]) \
                            and rngs[b].random() < dose:
                        fl[b, k] = True
        return fl
    return cb


def run(policy, source, cb, model, ep_seeds, seed):
    env = CoverageEnv(len(ep_seeds), with_ranges=True)
    o = env.reset(ep_seeds)
    gen = torch.Generator().manual_seed(seed)
    done = False
    while not done:
        a = heuristic_actions(env) if policy == "heuristic" else act(model, o, gen=gen)[0]
        o, _, done = env.step(a, source, cb)
    m = env.metrics()
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--conditions", nargs="+", default=[c[0] for c in CONDITIONS])
    args = ap.parse_args()
    acc = os.environ.get("ACC", "local")
    out_dir = Path(f"results/runs_{acc}/e8")
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_seeds = list(range(EVAL_SEED0, EVAL_SEED0 + args.episodes))
    print(f"Thiết bị: {DEV}", flush=True)
    t0 = time.time()
    log_score = logistic_detector() if any(c[3] == "logistic" for c in CONDITIONS if c[0] in args.conditions) else None
    print(f"  logistic sẵn sàng ({(time.time()-t0)/60:.1f} phút)", flush=True)
    for s in args.seeds:
        ts = time.time()
        model = ActorCritic()
        model.load_state_dict(torch.load(find(f"results/runs_*/e8/policy_s{s}.pt"), weights_only=False)["state"])
        dets = {}
        res = {"seed": s, "episodes": args.episodes, "conditions": {}}
        for name, pol, src, det, dose in CONDITIONS:
            if name not in args.conditions:
                continue
            cb = None
            if src == "callback":
                if det not in dets:
                    dets[det] = log_score if det == "logistic" else nn_detector(det, s)
                cb = make_callback(dets[det], dose, ep_seeds)
            res["conditions"][name] = run(pol, src, cb, model, ep_seeds, s)
            m = res["conditions"][name]
            f = np.array(m["has_fault"])
            print(f"  seed {s} | {name:18s} holes(lỗi) {np.mean(np.array(m['holes'])[f]):.4f} "
                  f"phủ(lỗi) {np.mean(np.array(m['coverage'])[f]):.4f} phủ(không lỗi) {np.mean(np.array(m['coverage'])[~f]):.4f} "
                  f"báo nhầm/mốc {np.mean(m['false_flag_rate']):.4f}", flush=True)
        res["minutes"] = (time.time() - ts) / 60
        json.dump(res, open(out_dir / f"eval_s{s}.json", "w"))
        print(f"-> {out_dir}/eval_s{s}.json ({res['minutes']:.1f} phút)", flush=True)
    print(f"Tổng: {(time.time()-t0)/60:.1f} phút")


if __name__ == "__main__":
    main()
