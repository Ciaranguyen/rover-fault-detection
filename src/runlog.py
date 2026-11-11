"""Ghi nhật ký mỗi lần chạy vào results/runs_<ACC>/runlog.csv (mỗi tài khoản một file riêng,
để hai tài khoản đẩy lên GitHub không bị xung đột). Tự ghi loại GPU và commit hiện tại.

Dùng trong notebook:
    import sys; sys.path.insert(0, "src")
    from runlog import log_run
    log_run("E0-gen", "E0", "-", "configs/data.yaml", "all", "-", 4.2, "data/", "sinh du lieu")
"""
import csv
import datetime
import os
import subprocess

HEADER = ["date", "run_id", "experiment", "model", "config", "data_split", "seed",
          "account", "gpu", "duration_min", "git_commit", "output_path", "notes"]


def _sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def log_run(run_id, experiment, model, config, data_split, seed, duration_min, output_path, notes=""):
    acc = os.environ.get("ACC", "local")
    gpu = _sh("nvidia-smi --query-gpu=name --format=csv,noheader") or "CPU"
    commit = _sh("git rev-parse --short HEAD")
    path = f"results/runs_{acc}/runlog.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow([datetime.date.today().isoformat(), run_id, experiment, model, config, data_split,
                    seed, acc, gpu, round(duration_min, 2), commit, output_path, notes])
    print(f"Đã ghi {run_id} vào {path}")
