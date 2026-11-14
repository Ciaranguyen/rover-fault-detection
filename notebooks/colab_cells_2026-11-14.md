# Ngày 14/11 — E3/E4/E5 (đánh giá cuối, chạy MỘT lần) và E6 (độ nhạy)

Pro: chạy src/final_eval.py trên test + 4 tập OOD cho cả 6 mô hình (checkpoint của Free cũng nằm trong repo).
Free: E6 — (a) GRU+GAT train bằng nhiễu Gaussian rồi đánh giá trên test nhiễu thật; (b) baseline với cửa sổ 10 và 60 bước (val).

Cell S giống ngày 13/11 (đổi ACC). Chạy `!git pull -q` trước.

## Pro
```python
# [14/11 | Pro | E3+E4+E5] Đánh giá cuối 6 mô hình trên test + 4 tập OOD, kèm KTC bootstrap — dự kiến 15–25 phút
!git pull -q
!ls results/runs_*/ckpt/ | head -30
t0 = time.time()
!python src/final_eval.py --splits test ood_drift8 ood_noise2 ood_two_faulty ood_n8 2>&1 | tee results/runs_$ACC/E3_final_log.txt
out = f"results/runs_{ACC}/E3_final_ood_n8.json"
if xong(out, t0):
    log_run("E3E4E5-final", "E3/E4/E5", "all", "ckpt v2, thr tu val", "test+ood", "10-14", (time.time()-t0)/60, f"results/runs_{ACC}/E3_final_*.json")
    push_git("E3/E4/E5: danh gia cuoi tren test va OOD (pro)")
else:
    print("CHƯA XONG — không ghi nhật ký, không đẩy")
```

## Free
```python
# [14/11 | Free | E6a] GRU+GAT train bằng nhiễu Gaussian (cùng cấu hình 64/3e-3), chọn trên val_gauss, 5 seed — dự kiến 20–30 phút
!git pull -q
t0 = time.time()
!python src/train_nn.py --model grugat --mode report --train_split train_gauss --val_split val_gauss --tag _gauss --hidden 64 --lr 0.003
out = f"results/runs_{ACC}/E2_grugat_gauss_report.json"
if xong(out, t0):
    log_run("E6-grugat-gauss-report", "E6", "grugat", "h64 lr3e-3, noise gaussian", "train_gauss/val_gauss", "10-14", (time.time()-t0)/60, out)
    push_git("E6: GRU+GAT train bang nhieu Gaussian (free)")
else:
    print("CHƯA XONG — dừng")
```
```python
# [14/11 | Free | E6a] Đánh giá mô hình train-Gaussian trên test nhiễu THẬT (so với GRU+GAT train nhiễu thật của Pro) — dự kiến 3 phút
t0 = time.time()
!python src/train_nn.py --model grugat --mode final --val_split val_gauss --tag _gauss --hidden 64 --lr 0.003 --splits test ood_noise2
out = f"results/runs_{ACC}/E3_grugat_gauss_final.json"
if xong(out, t0):
    log_run("E6-grugat-gauss-final", "E6", "grugat", "train gaussian, test empirical", "test,ood_noise2", "10-14", (time.time()-t0)/60, out)
    push_git("E6: danh gia mo hinh train Gaussian tren test (free)")
else:
    print("CHƯA XONG")
```
```python
# [14/11 | Free | E6b] Độ dài cửa sổ của baseline: 10 và 60 bước (trên val) — dự kiến 10–15 phút
for w in [10, 60]:
    for m in ["threshold", "logistic"]:
        t0 = time.time()
        !python src/baselines.py --model {m} --window {w}
        out = f"results/runs_{ACC}/E1_{m}_w{w}_val.json"
        if xong(out, t0):
            log_run(f"E6-{m}-w{w}", "E6", m, f"window {w}", "train/val", "-", (time.time()-t0)/60, out)
push_git("E6: baseline voi cua so 10 va 60 buoc (free)")
```
