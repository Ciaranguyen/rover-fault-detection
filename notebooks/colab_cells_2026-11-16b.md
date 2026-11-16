# 16/11 (tối) — E8: MARL phủ vùng + bộ phát hiện trực tuyến

Thiết kế và giả thuyết: docs/E8_preregistration.md (đã commit trước khi chạy).

## Bước 1 — train chính sách (song song). Không cần GPU (mạng rất nhỏ), nhưng để nguyên T4 cũng được.
Pro:
```python
# [16/11 | Pro | E8] Train MAPPO seed 10–19, 300 vòng mỗi seed — dự kiến 35–50 phút
t0 = time.time()
for s in range(10, 20):
    !python src/e8_train.py --seed {s} --iters 300 | tail -2
ok = all(os.path.exists(f"results/runs_{ACC}/e8/policy_s{s}.pt") for s in range(10, 20))
if ok:
    log_run("E8-train-10-19", "E8", "mappo", "300 iters x 64 env, trust noisy", "sim online", "10-19", (time.time()-t0)/60, f"results/runs_{ACC}/e8/")
    push_git("E8: train MAPPO seed 10-19 (pro)")
else:
    print("THIẾU seed — chạy lại cell")
```
Free: giống hệt, thay `range(10, 20)` bằng `range(20, 30)`, run_id "E8-train-20-29", lời commit "(free)".

## Bước 2 — đánh giá (mỗi tài khoản đánh giá seed của mình), cần GPU
```python
# [16/11 | Pro | E8] Đánh giá 10 điều kiện × 300 episode, seed 10–19, bộ phát hiện trực tuyến — dự kiến 25–40 phút
!git pull -q
t0 = time.time()
!python src/e8_eval.py --seeds 10 11 12 13 14 15 16 17 18 19 --episodes 300 2>&1 | tee results/runs_$ACC/E8_eval_log.txt
out = f"results/runs_{ACC}/e8/eval_s19.json"
if xong(out, t0):
    log_run("E8-eval-10-19", "E8", "all", "10 dieu kien, 300 ep", "sim online", "10-19", (time.time()-t0)/60, f"results/runs_{ACC}/e8/eval_s*.json")
    push_git("E8: danh gia seed 10-19 (pro)")
```
Free: thay danh sách seed bằng 20 … 29, `eval_s29.json`, run_id "E8-eval-20-29".

## Bước 3 — khi cả hai đã đẩy xong, chạy trên Pro
```python
# [16/11 | Pro | E8] Gộp 20 seed, so sánh ghép cặp, KTC hai tầng — dưới 1 phút
!git pull -q
!ls results/runs_*/e8/eval_s*.json | wc -l
!python src/e8_analyze.py 2>&1 | tee results/runs_$ACC/E8_summary.txt
push_git("E8: tong hop 20 seed (pro)")
```
