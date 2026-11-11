"""In mã băm SHA-256 của từng mảng trong các file .npz.
Dùng để kiểm tra hai tài khoản Colab sinh ra dữ liệu giống hệt nhau.

Chạy: python src/checksum.py data/train.npz data/val.npz data/test.npz
"""
import hashlib
import sys

import numpy as np

for path in sys.argv[1:]:
    z = np.load(path)
    h = hashlib.sha256()
    for k in sorted(z.files):
        h.update(k.encode())
        h.update(np.ascontiguousarray(z[k]).tobytes())
    print(f"{path}: {h.hexdigest()[:16]}")
