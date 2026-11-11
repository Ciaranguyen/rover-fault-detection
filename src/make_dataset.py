"""Sinh các tập dữ liệu theo configs/data.yaml.

Chạy:
    python src/make_dataset.py --split train val test        # các tập chính
    python src/make_dataset.py --split all                    # tất cả
    python src/make_dataset.py --split val --episodes 20      # chạy thử nhanh
Mỗi tập lưu thành data/<split>.npz (+ <split>_meta.json). Dữ liệu sinh lại được y hệt từ seed,
nên không cần đẩy lên GitHub: mỗi tài khoản Colab tự sinh.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import SimConfig, NoiseModel, simulate_episode, config_dict  # noqa: E402


def build_split(name, spec, noise_file, out_dir, episodes=None):
    cfg = SimConfig()
    ov = dict(spec.get("overrides", {}))
    for k in ("drift_rate", "t_fault"):
        if k in ov:
            ov[k] = tuple(ov[k])
    cfg = replace(cfg, **ov)
    noise = NoiseModel(noise_file, cfg)
    n_ep = episodes or spec["episodes"]
    eps, metas = [], []
    t0 = time.time()
    for e in range(n_ep):
        rng = np.random.default_rng([spec["seed_base"], e])  # seed xác định theo (tập, chỉ số episode)
        data, meta = simulate_episode(cfg, noise, rng)
        meta["seed"] = [spec["seed_base"], e]
        eps.append(data)
        metas.append(meta)
    arrays = {k: np.stack([d[k] for d in eps]) for k in eps[0]}
    out = Path(out_dir) / f"{name}.npz"
    np.savez_compressed(out, **arrays)
    with open(Path(out_dir) / f"{name}_meta.json", "w") as f:
        json.dump({"config": config_dict(cfg), "episodes": metas}, f)
    lab = arrays["label"]
    frac = {c: float((lab == c).mean()) for c in (0, 1, 2)}
    print(f"{name}: {n_ep} episode, {time.time()-t0:.1f}s, tỉ lệ nhãn (nút x bước) {frac}, -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--split", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--episodes", type=int, default=None, help="ghi đè số episode (để chạy thử)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    conf = yaml.safe_load(open(args.config))
    names = list(conf["splits"]) if args.split == ["all"] else args.split
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for n in names:
        build_split(n, conf["splits"][n], conf["noise_file"], args.out, args.episodes)


if __name__ == "__main__":
    main()
