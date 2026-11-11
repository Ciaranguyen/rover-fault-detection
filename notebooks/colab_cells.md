# Ngày 11/11 — các cell Colab

Thay `<GH_USER>` bằng tên GitHub, `<TEN>` và `<EMAIL>` bằng tên và email GitHub của bạn.
Tài khoản Pro chạy P1 → P9. Tài khoản miễn phí chạy F1 → F7.

## Tài khoản Pro

### P1 — kiểm tra GPU
```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

### P2 — token và tên tài khoản
```python
from google.colab import userdata
import os
os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN")
os.environ["ACC"] = "pro"
```

### P3 — clone repo
```python
!git clone -q https://x-access-token:$GH_TOKEN@github.com/<GH_USER>/rover-fault-detection.git
%cd /content/rover-fault-detection
!git config user.name "<TEN>"
!git config user.email "<EMAIL>"
!git log --oneline -1
```

### P4 — cài thư viện, ghi phiên bản
```python
!pip install -q -r requirements.txt
!mkdir -p results/runs_$ACC
!pip freeze > results/runs_$ACC/pip_freeze_2026-11-11.txt
!python -c "import numpy, sys; print('python', sys.version.split()[0], '| numpy', numpy.__version__)"
```

### P5 — sinh toàn bộ dữ liệu (khoảng 3–5 phút)
```python
import time
t0 = time.time()
!python src/make_dataset.py --split all
gen_min = (time.time() - t0) / 60
print(f"Sinh dữ liệu mất {gen_min:.1f} phút")
!ls -la data
```

### P6 — E0 và mã băm
```python
!python src/check_data.py data/val.npz | tee results/runs_$ACC/E0_check_val.txt
!python src/checksum.py data/train.npz data/val.npz data/test.npz | tee results/runs_$ACC/checksum.txt
```

### P7 — (chỉ Pro) kiểm tra tái tạo mô hình nhiễu từ dữ liệu gốc TIERS
```python
!git clone -q --depth 1 https://github.com/TIERS/uwb-relative-localization-dataset /content/tiers
t0 = time.time()
!python src/noise_tiers.py --tiers /content/tiers --out /content/noise_recheck.npz | tee results/runs_$ACC/noise_recheck.txt
noise_min = (time.time() - t0) / 60
import numpy as np
a, b = np.load("data/noise_tiers.npz"), np.load("/content/noise_recheck.npz")
same = all(np.allclose(a[k], b[k]) for k in a.files)
print("Mô hình nhiễu tái tạo GIỐNG bản trong repo:", same)
```

### P8 — ghi nhật ký
```python
import sys; sys.path.insert(0, "src")
from runlog import log_run
log_run("E0-gen-pro", "E0", "-", "configs/data.yaml", "all", "-", gen_min, "data/", "sinh du lieu + checksum")
log_run("E0-noise-pro", "E0", "-", "src/noise_tiers.py", "TIERS case3,4", "-", noise_min, "/content/noise_recheck.npz", f"tai tao nhieu giong repo: {same}")
```

### P9 — đẩy lên GitHub
```python
!git add results/runs_$ACC
!git commit -q -m "E0: sinh du lieu, checksum, kiem tra nhieu (pro)"
!git pull --rebase -q
!git push -q
!git log --oneline -3
```

## Tài khoản miễn phí

### F1 — kiểm tra GPU
Giống P1.

### F2 — token và tên tài khoản
Giống P2 nhưng `os.environ["ACC"] = "free"`.

### F3, F4, F5, F6
Giống P3, P4, P5, P6.

### F7 — ghi nhật ký và đẩy lên GitHub
```python
import sys; sys.path.insert(0, "src")
from runlog import log_run
log_run("E0-gen-free", "E0", "-", "configs/data.yaml", "all", "-", gen_min, "data/", "sinh du lieu + checksum")
!git add results/runs_$ACC
!git commit -q -m "E0: sinh du lieu, checksum (free)"
!git pull --rebase -q
!git push -q
```

## Sau khi cả hai đã đẩy — so sánh (chạy ở tài khoản nào cũng được)
```python
!git pull -q
!diff results/runs_pro/checksum.txt results/runs_free/checksum.txt && echo "TRÙNG: hai máy sinh dữ liệu giống hệt"
```
