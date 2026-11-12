# Ngày 13/11 — E1 (autoencoder) và E2 (GRU, GAT, GRU + GAT) trên val

Hôm nay chỉ dùng **train** và **val**. Không đụng tới test (để dành 14–15/11).

Thứ tự trong mỗi notebook: chạy cell S (đổi `ACC`), bật GPU (Runtime → Change runtime type → T4),
rồi chạy các cell của tài khoản đó theo thứ tự. Mỗi cell tự đẩy kết quả lên GitHub khi xong,
nên lỡ mất kết nối thì chỉ mất cell đang chạy.

Ngân sách dò siêu tham số bằng nhau cho mọi mô hình học: 4 cấu hình
(hidden ∈ {32, 64} × lr ∈ {1e-3, 3e-3}), tối đa 30 epoch, dừng sớm sau 5 epoch không cải thiện
macro-F1 trên val. Seed dò: 0. Seed báo cáo: 10–14.

### S — chuẩn bị (mọi lần mở notebook)
```python
# [13/11 | Pro + Free | S] Clone/pull repo, sinh lại dữ liệu, kiểm tra mã băm — dự kiến 5 phút
from google.colab import userdata
import os, time, sys, datetime
os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN")
os.environ["ACC"] = "pro"          # tài khoản free: đổi thành "free"
if not os.path.exists("/content/rover-fault-detection"):
    !git clone -q https://x-access-token:$GH_TOKEN@github.com/Ciaranguyen/rover-fault-detection.git /content/rover-fault-detection
%cd /content/rover-fault-detection
!git pull -q
!git config user.name "Ciara Nguyen"
!git config user.email "lamanh2003hd@gmail.com"
!git log --oneline -1
!pip install -q -r requirements.txt
!mkdir -p results/runs_$ACC
TODAY = datetime.date.today().isoformat()
t0 = time.time()
!python src/make_dataset.py --split all
gen_min = (time.time() - t0) / 60
!python src/checksum.py data/train.npz data/val.npz data/test.npz | tee results/runs_$ACC/checksum_{TODAY}.txt
sys.path.insert(0, "src")
from runlog import log_run
!nvidia-smi --query-gpu=name --format=csv,noheader || echo "KHÔNG có GPU"

ACC = os.environ["ACC"]
def xong(path, t_start):
    """Chỉ ghi nhật ký khi file kết quả được tạo SAU lúc bắt đầu chạy (tránh ghi lần chạy hỏng)."""
    return os.path.exists(path) and os.path.getmtime(path) > t_start
def push_git(msg):
    !git add results/runs_$ACC
    !git commit -q -m "{msg}"
    !git pull --rebase -q
    !git push -q
```
Mã băm đúng: train `26decb9df0553ac7`, val `f6a32b4b58528bca`, test `b972f3f1d379bb76`.

## Tài khoản Pro — mô hình chính GRU + GAT

### P13-1 — dò siêu tham số GRU + GAT
```python
# [13/11 | Pro | E2] Dò 4 cấu hình GRU+GAT trên val (seed 0) — dự kiến 1–2 giờ
t0 = time.time()
!python src/train_nn.py --model grugat --mode tune
out = f"results/runs_{ACC}/E2_grugat_tune.json"
if xong(out, t0):
    log_run("E2-grugat-tune", "E2", "grugat", "grid h{32,64} x lr{1e-3,3e-3}", "train/val", 0, (time.time()-t0)/60, out)
    push_git("E2: do sieu tham so GRU+GAT (pro)")
else:
    print("CHƯA XONG — không ghi nhật ký, không đẩy")
```

### P13-2 — train 5 seed báo cáo GRU + GAT
```python
# [13/11 | Pro | E2] Train lại cấu hình tốt nhất với seed 10–14, lưu checkpoint — dự kiến 1,5–2,5 giờ
t0 = time.time()
!python src/train_nn.py --model grugat --mode report
out = f"results/runs_{ACC}/E2_grugat_report.json"
if xong(out, t0):
    log_run("E2-grugat-report", "E2", "grugat", "best of tune", "train/val", "10-14", (time.time()-t0)/60, out)
    push_git("E2: GRU+GAT 5 seed bao cao + checkpoint (pro)")
else:
    print("CHƯA XONG — không ghi nhật ký, không đẩy")
```

## Tài khoản Free — autoencoder và hai bản ablation

Ba mô hình chạy cùng một hàm, mỗi mô hình là một cell (tune rồi report liền nhau).

```python
# [13/11 | Free | E1/E2] Hàm dùng chung cho 3 cell bên dưới — chạy 1 lần, vài giây
def tune_report(model, exp):
    for mode in ["tune", "report"]:
        t0 = time.time()
        !python src/train_nn.py --model {model} --mode {mode}
        out = f"results/runs_{ACC}/E2_{model}_{mode}.json"
        if not xong(out, t0):
            print(f"{model} {mode} CHƯA XONG — dừng"); return
        seed = 0 if mode == "tune" else "10-14"
        log_run(f"{exp}-{model}-{mode}", exp, model, "grid h{32,64} x lr{1e-3,3e-3}" if mode == "tune" else "best of tune",
                "train/val", seed, (time.time()-t0)/60, out)
        push_git(f"{exp}: {model} {mode} (free)")
```

### F13-1 — autoencoder (baseline 3)
```python
# [13/11 | Free | E1] Autoencoder: dò 4 cấu hình + 5 seed báo cáo — dự kiến 20–40 phút
tune_report("ae", "E1")
```

### F13-2 — chỉ GRU (ablation, chỉ thời gian)
```python
# [13/11 | Free | E2] GRU-only: dò 4 cấu hình + 5 seed báo cáo — dự kiến 30–60 phút
tune_report("gru", "E2")
```

### F13-3 — chỉ GAT (ablation, chỉ đồ thị)
```python
# [13/11 | Free | E2] GAT-only: dò 4 cấu hình + 5 seed báo cáo — dự kiến 1–2 giờ
tune_report("gat", "E2")
```

### Cuối ngày (cả hai tài khoản) — xem nhanh kết quả val
```python
# [13/11 | Pro + Free | E2] In bảng tóm tắt val của mọi mô hình đã có — vài giây
!git pull -q
import json, glob
for f in sorted(glob.glob("results/runs_*/E2_*_report.json")):
    s = json.load(open(f))["summary_val"]
    print(f"{f:45s} macroF1 {s['macro_f1']['mean']:.4f}±{s['macro_f1']['std']:.4f}  "
          f"recall@1% {s['recall_fault_at_thr']['mean']:.4f}  delay {s['median_delay_steps']['mean']:.1f}")
```
Gửi lại output cell này cho Claude.

Nếu free bị hết hạn mức GPU: vẫn chạy được trên CPU (AE và GRU nhanh), riêng GAT sẽ chậm
khoảng 5–10 lần — khi đó để GAT sang sáng 14/11.
