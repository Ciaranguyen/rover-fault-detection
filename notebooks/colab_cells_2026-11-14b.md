# Ngày 14/11 (tối) — E7: so sánh công bằng + biến thể train thêm 2 rover lỗi (_mf)

Cell S như cũ (sinh thêm 4 tập mới: val_two_faulty, val_noise2, ood_two_faulty_b, train_two).

## Bước 1 — train biến thể _mf (song song)
Pro:
```python
# [14/11 tối | Pro | E7] GRU+GAT_mf: train trên train + train_two (1000 ep 2 rover lỗi), cấu hình cũ 64/3e-3, 5 seed — dự kiến 25–35 phút
t0 = time.time()
!python src/train_nn.py --model grugat --mode report --train_split train train_two --tag _mf --hidden 64 --lr 0.003
out = f"results/runs_{ACC}/E2_grugat_mf_report.json"
if xong(out, t0):
    log_run("E7-grugat-mf", "E7", "grugat_mf", "h64 lr3e-3, train+train_two", "train+train_two/val", "10-14", (time.time()-t0)/60, out)
    push_git("E7: GRU+GAT train them 2 rover loi (pro)")
else:
    print("CHƯA XONG")
```
Free:
```python
# [14/11 tối | Free | E7] GRU_mf: train trên train + train_two, cấu hình cũ 64/3e-3, 5 seed — dự kiến 15–20 phút
t0 = time.time()
!python src/train_nn.py --model gru --mode report --train_split train train_two --tag _mf --hidden 64 --lr 0.003
out = f"results/runs_{ACC}/E2_gru_mf_report.json"
if xong(out, t0):
    log_run("E7-gru-mf", "E7", "gru_mf", "h64 lr3e-3, train+train_two", "train+train_two/val", "10-14", (time.time()-t0)/60, out)
    push_git("E7: GRU train them 2 rover loi (free)")
else:
    print("CHƯA XONG")
```

## Bước 2 — khi CẢ HAI đã đẩy xong, chạy trên Pro
```python
# [14/11 tối | Pro | E7] So sánh công bằng: AUROC, ngưỡng val cùng điều kiện, oracle; mọi mô hình kể cả _mf, _gauss — dự kiến 10–15 phút
!git pull -q
t0 = time.time()
!python src/fair_eval.py --pairs test:val ood_two_faulty:val_two_faulty ood_two_faulty_b:val_two_faulty ood_noise2:val_noise2 ood_n8:- ood_drift8:- 2>&1 | tee results/runs_$ACC/E7_fair_log.txt
out = f"results/runs_{ACC}/E7_fair_ood_drift8.json"
if xong(out, t0):
    log_run("E7-fair", "E7", "all", "AUROC + thr val cung dk + oracle", "test+ood", "10-14", (time.time()-t0)/60, f"results/runs_{ACC}/E7_fair_*.json")
    push_git("E7: so sanh cong bang (pro)")
```
