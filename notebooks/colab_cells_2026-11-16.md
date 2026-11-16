# Ngày 16/11 — tăng lên 20 seed (power analysis) cho GRU, GRU+GAT, GAT

Lý do: với 5 seed, nửa độ rộng KTC 95% của chênh lệch giữa hai mô hình (~0.029) lớn hơn chính chênh lệch
GRU+GAT − GRU (~0.02). Power analysis: cần ~15–20 seed mỗi mô hình. Cấu hình GIỮ NGUYÊN (64/3e-3), không dò lại.
Seed mới: 15–29. Checkpoint mỗi tài khoản lưu ở runs_<ACC>/ckpt, final_eval tìm ở mọi runs_*.

## Pro
```python
# [16/11 | Pro | E2-ext] GRU+GAT thêm 15 seed (15–29), cấu hình cũ — dự kiến 45–60 phút
t0 = time.time()
!python src/train_nn.py --model grugat --mode report --hidden 64 --lr 0.003 --seeds 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
out = f"results/runs_{ACC}/E2_grugat_report_s15-29.json"
if xong(out, t0):
    log_run("E2-grugat-ext", "E2", "grugat", "h64 lr3e-3", "train/val", "15-29", (time.time()-t0)/60, out)
    push_git("E2: GRU+GAT them seed 15-29 (pro)")
else:
    print("CHƯA XONG")
```

## Free
```python
# [16/11 | Free | E2-ext] GRU rồi GAT, mỗi mô hình thêm 15 seed (15–29), cấu hình cũ — dự kiến 50–70 phút
for m in ["gru", "gat"]:
    t0 = time.time()
    !python src/train_nn.py --model {m} --mode report --hidden 64 --lr 0.003 --seeds 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
    out = f"results/runs_{ACC}/E2_{m}_report_s15-29.json"
    if xong(out, t0):
        log_run(f"E2-{m}-ext", "E2", m, "h64 lr3e-3", "train/val", "15-29", (time.time()-t0)/60, out)
        push_git(f"E2: {m} them seed 15-29 (free)")
    else:
        print(f"{m} CHƯA XONG"); break
```

## Khi CẢ HAI đã đẩy xong — Pro: đánh giá lại với 20 seed, bootstrap hai tầng
```python
# [16/11 | Pro | E3+E7 v2] Đánh giá cuối + so sánh công bằng với 20 seed, KTC hai tầng (seed × episode) — dự kiến 25–35 phút
!git pull -q
!ls results/runs_*/ckpt/ | grep -c pt
!mkdir -p results/runs_$ACC/v1_5seed && cp results/runs_$ACC/E3_final_*.json results/runs_$ACC/E7_fair_*.json results/runs_$ACC/v1_5seed/ 2>/dev/null; ls results/runs_$ACC/v1_5seed
t0 = time.time()
!python src/final_eval.py --splits test ood_drift8 ood_noise2 ood_two_faulty ood_n8 2>&1 | tee results/runs_$ACC/E3_final_log_v2.txt
!python src/fair_eval.py --pairs test:val ood_two_faulty:val_two_faulty ood_two_faulty_b:val_two_faulty ood_noise2:val_noise2 ood_n8:- ood_drift8:- 2>&1 | tee results/runs_$ACC/E7_fair_log_v2.txt
out = f"results/runs_{ACC}/E7_fair_ood_drift8.json"
if xong(out, t0):
    log_run("E3E7-v2-20seed", "E3/E7", "all", "20 seed, bootstrap 2 tang", "test+ood", "10-29", (time.time()-t0)/60, "E3_final_*.json, E7_fair_*.json")
    push_git("E3/E7 v2: 20 seed, KTC hai tang (pro)")
```

## Đối chứng cho _mf (thêm 16/11) — Free, sau khi xong GRU/GAT thêm seed
```python
# [16/11 | Free | E7-ctrl] GRU+GAT_more: train + 1000 ep CHỈ 1 rover lỗi (cùng lượng dữ liệu với _mf), 5 seed — dự kiến 25–35 phút
t0 = time.time()
!python src/train_nn.py --model grugat --mode report --train_split train train_extra --tag _more --hidden 64 --lr 0.003
out = f"results/runs_{ACC}/E2_grugat_more_report.json"
if xong(out, t0):
    log_run("E7-grugat-more", "E7", "grugat_more", "h64 lr3e-3, train+train_extra", "train+train_extra/val", "10-14", (time.time()-t0)/60, out)
    push_git("E7: doi chung GRU+GAT them du lieu 1 rover loi (free)")
else:
    print("CHƯA XONG")
```
Số checkpoint mong đợi trước bước đánh giá v2: `grep -c pt` = 87 (85 file .pt + 2 dòng tên thư mục).
