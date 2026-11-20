# Ngày 12/11 — các cell Colab

Hôm nay mô phỏng được sửa: nhiễu UWB giờ giữ **tương quan theo thời gian** như dữ liệu thật
(ghép các đoạn 20 giây liên tiếp của chuỗi nhiễu TIERS). Dữ liệu và mã băm vì vậy khác hôm qua,
nên cả hai tài khoản phải sinh lại.

Mỗi lần mở notebook (máy ảo mới), chạy cell S trước. Đổi `ACC` thành `"pro"` hoặc `"free"`.

### S — chuẩn bị (mọi lần mở notebook, khoảng 5 phút)
```python
from google.colab import userdata
import os, time, sys
os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN")
os.environ["ACC"] = "free"          # hoặc "pro"
if not os.path.exists("/content/rover-fault-detection"):
    !git clone -q https://x-access-token:$GH_TOKEN@github.com/Ciaranguyen/rover-fault-detection.git /content/rover-fault-detection
%cd /content/rover-fault-detection
!git pull -q
!git config user.name "Ciara Nguyen"
!git config user.email "<EMAIL>"
!git log --oneline -1
!pip install -q -r requirements.txt
!mkdir -p results/runs_$ACC
t0 = time.time()
!python src/make_dataset.py --split all
gen_min = (time.time() - t0) / 60
!python src/checksum.py data/train.npz data/val.npz data/test.npz | tee results/runs_$ACC/checksum_2026-11-12.txt
sys.path.insert(0, "src")
from runlog import log_run
```
Mã băm đúng (khớp máy của Claude):
```
data/train.npz: 26decb9df0553ac7
data/val.npz:   f6a32b4b58528bca
data/test.npz:  b972f3f1d379bb76
```

## Tài khoản Pro

### P-a — kiểm tra lại mô hình nhiễu mới từ dữ liệu gốc TIERS
```python
!git clone -q --depth 1 https://github.com/TIERS/uwb-relative-localization-dataset /content/tiers 2>/dev/null || true
t0 = time.time()
!python src/noise_tiers.py --tiers /content/tiers --out /content/noise_recheck.npz | tee results/runs_$ACC/noise_recheck_2026-11-12.txt
import numpy as np
a, b = np.load("data/noise_tiers.npz"), np.load("/content/noise_recheck.npz")
same = all(np.allclose(a[k], b[k]) for k in a.files)
print("Mô hình nhiễu tái tạo GIỐNG bản trong repo:", same)
log_run("E0-gen-pro-v2", "E0", "-", "configs/data.yaml", "all", "-", gen_min, "data/", "nhieu block 20s")
log_run("E0-noise-pro-v2", "E0", "-", "src/noise_tiers.py", "TIERS case3,4", "-", (time.time()-t0)/60, "-", f"giong repo: {same}")
!python src/check_data.py data/val.npz | tee results/runs_$ACC/E0_check_val_v2.txt
```

### P-b — đẩy lên GitHub
```python
!git add results/runs_$ACC
!git commit -q -m "E0 v2: nhieu tuong quan thoi gian (pro)"
!git pull --rebase -q
!git push -q
```

## Tài khoản miễn phí (chạy CPU, không cần GPU)

### F-a — E1: hai baseline, dò trên tập val
```python
t0 = time.time()
!python src/baselines.py --model threshold
log_run("E1-threshold-val", "E1", "threshold", "w30; t1 @FPR1%, t2 max F1", "val", "-", (time.time()-t0)/60, "results/runs_free/E1_threshold_w30_val.json")
t0 = time.time()
!python src/baselines.py --model logistic
log_run("E1-logistic-val", "E1", "logistic", "w30; C in {0.01,0.1,1,10}", "train/val", "-", (time.time()-t0)/60, "results/runs_free/E1_logistic_w30_val.json")
```
Kết quả nên gần (trên val):
```
E1_threshold_w30: macro-F1 0.4622 | recall lỗi @ báo nhầm<=1%: 0.1681 ...
E1_logistic_w30:  macro-F1 0.6560 | recall lỗi @ báo nhầm<=1%: 0.3459 ...
```

### F-b — đẩy lên GitHub
```python
!git add results/runs_$ACC
!git commit -q -m "E1: baseline nguong va logistic tren val (free)"
!git pull --rebase -q
!git push -q
```

Lưu ý: KHÔNG chạy `--final` hôm nay. Tập test chỉ chạy một lần vào ngày 14/11, sau khi mọi mô hình đã chốt cấu hình.
