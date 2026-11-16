"""E8: huấn luyện chính sách phân khu bằng MAPPO (tham số chung cho mọi rover, critic tập trung).

Chính sách được huấn luyện với nguồn tin cậy 'noisy' (oracle có trễ, bỏ sót, báo nhầm ngẫu nhiên), KHÔNG dùng
bộ phát hiện nào cụ thể — để khi đánh giá (e8_eval.py) có thể cắm từng bộ phát hiện vào CÙNG một chính sách
(thiết kế ghép cặp). Episode huấn luyện dùng seed 0..; episode đánh giá dùng seed từ 10^6, không trùng.

Chạy:  python src/e8_train.py --seed 10 --iters 300
Ghi:   results/runs_<ACC>/e8/policy_s<seed>.pt, results/runs_<ACC>/e8/train_s<seed>.json (đường cong học)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e8_env import GLOBAL_EXTRA, N_DEC, OBS_DIM, R, CoverageEnv, heuristic_actions  # noqa: E402


class ActorCritic(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(OBS_DIM, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, R))
        self.critic = nn.Sequential(nn.Linear(OBS_DIM + GLOBAL_EXTRA, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                                    nn.Linear(h, 1))

    def dist(self, o):
        return torch.distributions.Categorical(logits=self.actor(o))

    def value(self, o, g):
        return self.critic(torch.cat([o, g], -1)).squeeze(-1)


def act(model, obs, greedy=False, gen=None):
    """gen: torch.Generator cố định khi đánh giá -> mọi điều kiện dùng cùng chuỗi số ngẫu nhiên khi lấy mẫu hành động."""
    with torch.no_grad():
        d = model.dist(torch.from_numpy(obs))
        if greedy:
            a = d.probs.argmax(-1)
        elif gen is not None:
            a = torch.multinomial(d.probs.reshape(-1, d.probs.shape[-1]), 1, generator=gen).reshape(d.probs.shape[:-1])
        else:
            a = d.sample()
        return a.numpy(), d.log_prob(a).numpy()


def rollout(env, model, seeds, source="noisy"):
    o = env.reset(seeds)
    B, N = env.B, env.N
    buf = {k: [] for k in ["obs", "glob", "act", "logp", "val", "rew"]}
    done = False
    while not done:
        g = np.repeat(env.global_extra()[:, None], N, 1)
        a, lp = act(model, o)
        with torch.no_grad():
            v = model.value(torch.from_numpy(o), torch.from_numpy(g)).numpy()
        o2, r, done = env.step(a, source)
        for k, x in zip(buf, [o, g, a, lp, v, np.repeat(r[:, None], N, 1)]):
            buf[k].append(x)
        o = o2
    out = {k: np.stack(v, 1) for k, v in buf.items()}            # (B, N_DEC, N, ...)
    return out, env.metrics()


def gae(rew, val, gamma=0.99, lam=0.95):
    B, Tn, N = rew.shape
    adv = np.zeros_like(rew)
    last = np.zeros((B, N))
    for t in reversed(range(Tn)):
        nv = val[:, t + 1] if t + 1 < Tn else 0.0
        delta = rew[:, t] + gamma * nv - val[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    args = ap.parse_args()
    acc = os.environ.get("ACC", "local")
    out_dir = Path(f"results/runs_{acc}/e8")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)

    env = CoverageEnv(args.envs)
    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    # tham chiếu: chính sách heuristic trên cùng kiểu episode huấn luyện
    o = env.reset(rng.integers(0, 10 ** 6, args.envs))
    done = False
    while not done:
        o, _, done = env.step(heuristic_actions(env), "noisy")
    heur = float(env.metrics()["coverage"].mean())
    print(f"Heuristic (episode huấn luyện): độ phủ thật {heur:.3f}", flush=True)

    curve, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        buf, m = rollout(env, model, rng.integers(0, 10 ** 6, args.envs))
        adv, ret = gae(buf["rew"], buf["val"])
        flat = lambda x: torch.from_numpy(x.reshape(-1, *x.shape[3:]))
        O, Gx, A, LP, ADV, RET = map(flat, [buf["obs"], buf["glob"], buf["act"], buf["logp"],
                                             adv.astype(np.float32), ret.astype(np.float32)])
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        M = len(A)
        for _ in range(4):
            perm = torch.randperm(M)
            for s in range(0, M, M // 4):
                i = perm[s:s + M // 4]
                d = model.dist(O[i])
                ratio = torch.exp(d.log_prob(A[i]) - LP[i])
                pl = -torch.min(ratio * ADV[i], torch.clamp(ratio, 0.8, 1.2) * ADV[i]).mean()
                vl = ((model.value(O[i], Gx[i]) - RET[i]) ** 2).mean()
                loss = pl + 0.5 * vl - args.ent * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        cov = float(m["coverage"].mean())
        curve.append({"iter": it, "coverage": cov, "holes": float(m["holes"].mean())})
        if it % 10 == 0 or it == 1:
            last = np.mean([c["coverage"] for c in curve[-10:]])
            print(f"  iter {it:4d}  độ phủ thật (TB 10 lần gần nhất) {last:.3f}  | heuristic {heur:.3f}  "
                  f"({(time.time()-t0)/60:.1f} phút)", flush=True)
    torch.save({"state": model.state_dict(), "seed": args.seed, "iters": args.iters}, out_dir / f"policy_s{args.seed}.pt")
    json.dump({"seed": args.seed, "heuristic_train_cov": heur, "curve": curve,
               "minutes": (time.time() - t0) / 60}, open(out_dir / f"train_s{args.seed}.json", "w"))
    print(f"-> {out_dir}/policy_s{args.seed}.pt")


if __name__ == "__main__":
    main()
