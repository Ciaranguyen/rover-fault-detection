"""E8 (chẩn đoán, thêm 17/11 SAU khi xem kết quả): vì sao báo nhầm trực tuyến cao hơn nhiều so với 1% ở E3?

Ở E8, tỉ lệ rover trung thực bị gắn cờ mỗi mốc là ~10% (GRU+GAT) và ~21% (GRU) khi đi với chính sách MAPPO,
nhưng chỉ ~1,6% khi GRU+GAT đi với chính sách heuristic. Giả thuyết: chuyển dịch phân phối do cách di chuyển —
dữ liệu train của bộ phát hiện chỉ có rover luôn chạy (quét làn), còn chính sách MAPPO phủ gần xong sớm nên nhiều
rover đứng yên hoặc đổi hướng liên tục.

Script ghi lại, ở mỗi mốc kiểm tra, với từng rover TRUNG THỰC: có bị gắn cờ không, tốc độ tự khai trung bình
20 bước gần nhất, thời điểm, số rover đang đo được tới nó. Rồi tách tỉ lệ báo nhầm theo các nhóm đó.

Chạy:  python src/e8_diag.py --seeds 10 11 12 --episodes 300
Ghi:   results/runs_<ACC>/e8/E8_diag.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e8_env import CoverageEnv, heuristic_actions  # noqa: E402
from e8_eval import EVAL_SEED0, find, nn_detector  # noqa: E402
from e8_train import ActorCritic, act  # noqa: E402


def run(policy, det_name, seed, ep_seeds, model):
    score_fn = nn_detector(det_name, seed)
    rec = []

    def cb(env):
        sc, thr = score_fn(env.data_prefix())
        fl = sc >= thr
        t = env.t
        c = env.claims[:, max(t - 20, 0):t]                              # (B,<=20,N,2)
        spd = np.linalg.norm(np.diff(c, axis=1), axis=-1).mean(1)        # (B,N) tốc độ tự khai TB
        nb = np.isfinite(env.r[:, t - 1]).sum(1)                         # (B,N) số rover đo được tới k
        is_f = (np.arange(env.N)[None] == env.faulty[:, None]) & (t >= env.t_u)[:, None]
        for b, k in zip(*np.where(~is_f)):
            rec.append((t, float(spd[b, k]), int(nb[b, k]), bool(fl[b, k])))
        return fl

    env = CoverageEnv(len(ep_seeds), with_ranges=True)
    o = env.reset(ep_seeds)
    gen = torch.Generator().manual_seed(seed)
    done = False
    while not done:
        a = heuristic_actions(env) if policy == "heuristic" else act(model, o, gen=gen)[0]
        o, _, done = env.step(a, "callback", cb)
    return np.array(rec, dtype=float)


def table(R):
    t, spd, nb, fl = R.T
    out = {"all": float(fl.mean())}
    for lo, hi in [(0, 0.05), (0.05, 0.2), (0.2, 0.28), (0.28, 1.0)]:
        m = (spd >= lo) & (spd < hi)
        out[f"tốc độ {lo}-{hi} m/s"] = [float(fl[m].mean()) if m.any() else None, float(m.mean())]
    for lo, hi in [(30, 150), (150, 250), (250, 400)]:
        m = (t >= lo) & (t < hi)
        out[f"bước {lo}-{hi}"] = [float(fl[m].mean()) if m.any() else None, float(m.mean())]
    for v in [0, 1, 2]:
        m = nb == v if v < 2 else nb >= 2
        out[f"{v}{'+' if v == 2 else ''} người quan sát"] = [float(fl[m].mean()) if m.any() else None, float(m.mean())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--episodes", type=int, default=300)
    args = ap.parse_args()
    acc = os.environ.get("ACC", "local")
    ep_seeds = list(range(EVAL_SEED0, EVAL_SEED0 + args.episodes))
    res = {}
    for pol in ["marl", "heuristic"]:
        for det in ["gru", "grugat"]:
            Rs = []
            for s in args.seeds:
                model = ActorCritic()
                model.load_state_dict(torch.load(find(f"results/runs_*/e8/policy_s{s}.pt"), weights_only=False)["state"])
                Rs.append(run(pol, det, s, ep_seeds, model))
            res[f"{pol}+{det}"] = table(np.concatenate(Rs))
            print(f"\n{pol}+{det}: báo nhầm/mốc chung {res[f'{pol}+{det}']['all']:.4f}")
            for k, v in res[f"{pol}+{det}"].items():
                if k != "all":
                    print(f"   {k:22s} báo nhầm {v[0] if v[0] is None else round(v[0], 4)}   (tỉ trọng mẫu {v[1]:.3f})")
    path = Path(f"results/runs_{acc}/e8/E8_diag.json")
    json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
