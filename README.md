# Phát hiện robot báo sai vị trí bằng mạng nơ-ron đồ thị theo thời gian

Mã nguồn bài tập lớn môn Học máy nâng cao (Trường CNTT Phenikaa, 2026).

Một đội 5 rover không có GPS: mỗi rover tự khai vị trí và đo khoảng cách UWB tới đồng đội. Tại mỗi giây,
mô hình phân loại từng rover thành **trung thực / trôi định vị (drift) / khai giả (spoof)**, chỉ dùng dữ liệu
đến thời điểm hiện tại. Mô hình chính là **GRU+GAT** (GRU theo thời gian + GATv2 trên đồ thị phép đo), so
với baseline ngưỡng, hồi quy logistic, autoencoder và hai biến thể ablation (GRU-only, GAT-only). Phần cuối
đưa bộ phát hiện vào một nhiệm vụ phủ vùng do chính sách MAPPO điều khiển (E8).

## Kết quả chính (tập kiểm tra, 20 seed)

| Mô hình | Macro-F1 | Recall khi báo nhầm 1% |
|---|---|---|
| GRU+GAT | 0,883 ± 0,017 | 0,872 ± 0,023 |
| GRU-only | 0,865 ± 0,019 | 0,839 ± 0,019 |
| GAT-only | 0,718 | 0,715 |
| Hồi quy logistic | 0,661 | 0,350 |
| Autoencoder | 0,588 | 0,377 |
| Ngưỡng | 0,471 | 0,181 |

Phân tích đầy đủ (độ bền, so sánh công bằng, E8, các hạn chế) nằm trong báo cáo.

## Cấu trúc thư mục

```
configs/
  data.yaml            các tập dữ liệu, mỗi tập một khoảng seed riêng (không giao nhau)
  seeds.yaml           seed dò siêu tham số (0) và seed báo cáo (10–29)
data/
  noise_tiers.npz      mô hình nhiễu UWB trích từ dữ liệu thật TIERS (các tập khác sinh lại từ seed)
src/
  noise_tiers.py       trích nhiễu từ rosbag TIERS
  sim.py               mô phỏng đội rover, định vị tự khai, đo UWB, ba loại lỗi
  make_dataset.py      sinh các tập dữ liệu -> data/<split>.npz
  checksum.py          mã băm SHA-256 để đối chiếu dữ liệu giữa các máy
  check_data.py        E0: phân bố phần dư theo nhóm
  features.py          đặc trưng cửa sổ nhân quả cho baseline
  baselines.py         E1, E6: baseline ngưỡng và hồi quy logistic
  tensors.py           đầu vào dạng đồ thị theo thời gian cho mạng nơ-ron
  models.py            GRU-only, GAT-only, GRU+GAT, autoencoder; lớp GATv2 tự cài
  train_nn.py          E1 (autoencoder), E2: dò siêu tham số, huấn luyện theo seed
  metrics.py           macro-F1, recall ở FPR 1%, độ trễ phát hiện, ECE
  final_eval.py        E3–E5: đánh giá cuối, phân tích theo nhóm, bootstrap hai tầng
  fair_eval.py         E7: AUROC, ngưỡng hiệu chỉnh cùng điều kiện, biến thể _mf/_more
  e8_env.py            E8: môi trường phủ vùng 16×16 ô
  e8_train.py          E8: huấn luyện MAPPO
  e8_eval.py           E8: đánh giá ghép cặp 10 điều kiện
  e8_analyze.py        E8: tương phản và khoảng tin cậy
  e8_diag.py           E8: chẩn đoán báo nhầm (phân tích sau khi xem kết quả)
  runlog.py            ghi nhật ký mỗi lần chạy
docs/
  E8_preregistration.md   giả thuyết E8, commit trước khi chạy đánh giá (ad23cf8)
notebooks/
  colab_cells_*.md     các cell Colab đã chạy, theo ngày
results/
  runs_pro/, runs_free/   kết quả JSON, log, runlog.csv, pip_freeze, mã băm dữ liệu
    ckpt/                 trọng số mạng nơ-ron: E2_<mô hình><biến thể>_s<seed>.pt
    e8/                   chính sách MAPPO và kết quả E8
```

## Cài đặt

```bash
git clone https://github.com/Ciaranguyen/rover-fault-detection.git
cd rover-fault-detection
pip install -r requirements.txt
```

Thí nghiệm gốc chạy trên Google Colab (GPU T4). Mọi lệnh chạy từ thư mục gốc của repo. Biến môi trường
`ACC` chọn thư mục ghi kết quả (`results/runs_$ACC`), ví dụ `export ACC=local`.

## Tái hiện thí nghiệm

### 1. Dữ liệu

```bash
python src/make_dataset.py --split all          # vài phút trên CPU
python src/checksum.py data/train.npz data/val.npz data/test.npz
```
So mã băm với `results/runs_pro/checksum.txt` để chắc dữ liệu sinh lại trùng khớp.

Trích lại mô hình nhiễu từ dữ liệu gốc (không bắt buộc, `data/noise_tiers.npz` đã có sẵn):
```bash
git clone --depth 1 https://github.com/TIERS/uwb-relative-localization-dataset ../tiers
python src/noise_tiers.py --tiers ../tiers --out data/noise_tiers.npz
```

### 2. E0–E1: hiểu dữ liệu và baseline

```bash
python src/check_data.py data/val.npz
python src/baselines.py --model threshold
python src/baselines.py --model logistic
python src/train_nn.py --model ae --mode tune
python src/train_nn.py --model ae --mode report --hidden 64 --lr 0.003
```

### 3. E2: dò siêu tham số và huấn luyện (chỉ dùng train và val)

```bash
for m in grugat gru gat; do
  python src/train_nn.py --model $m --mode tune            # 4 cấu hình, seed 0
  python src/train_nn.py --model $m --mode report --hidden 64 --lr 0.003 \
      --seeds 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
done
```

### 4. E3–E5, E7: đánh giá cuối (chạy một lần, sau khi đã chốt cấu hình)

```bash
python src/final_eval.py --splits test ood_drift8 ood_noise2 ood_two_faulty ood_n8
python src/train_nn.py --model grugat --mode report --train_split train train_two   --tag _mf   --hidden 64 --lr 0.003
python src/train_nn.py --model gru    --mode report --train_split train train_two   --tag _mf   --hidden 64 --lr 0.003
python src/train_nn.py --model grugat --mode report --train_split train train_extra --tag _more --hidden 64 --lr 0.003
python src/fair_eval.py --pairs test:val ood_two_faulty:val_two_faulty ood_two_faulty_b:val_two_faulty \
    ood_noise2:val_noise2 ood_n8:- ood_drift8:-
```

Có thể bỏ qua bước huấn luyện và chạy thẳng đánh giá: `final_eval.py` và `fair_eval.py` tự tìm checkpoint
trong `results/runs_*/ckpt`.

### 5. E6: độ nhạy

```bash
for w in 10 60; do
  python src/baselines.py --model threshold --window $w
  python src/baselines.py --model logistic  --window $w
done
python src/train_nn.py --model grugat --mode report --train_split train_gauss --val_split val_gauss --tag _gauss --hidden 64 --lr 0.003
python src/train_nn.py --model grugat --mode final  --val_split val_gauss --tag _gauss --hidden 64 --lr 0.003 --splits test ood_noise2
```

### 6. E8: ứng dụng vào phối hợp đa robot

```bash
for s in $(seq 10 29); do python src/e8_train.py --seed $s --iters 300; done
python src/e8_eval.py --seeds $(seq 10 29) --episodes 300
python src/e8_analyze.py
python src/e8_diag.py --seeds 10 11 12 --episodes 300
```

## Thời gian chạy

Tổng các lần chạy thành công trong `runlog.csv` khoảng 10,4 giờ trên GPU T4 và CPU của Colab. Một seed
GRU+GAT mất khoảng 3,2 phút, GRU-only 0,5 phút, GAT-only 2,5 phút; E8 khoảng 3,5 giờ.

## Quy tắc thực nghiệm

- Các tập chia theo episode, sinh từ khoảng seed không giao nhau.
- Siêu tham số, epoch và ngưỡng báo động chỉ chọn trên train/val. Tập test và `ood_*` chạy sau khi chốt cấu hình.
- Vị trí thật và nhãn không bao giờ là đầu vào của mô hình.
- Các tập và phân tích thêm vào sau khi đã xem kết quả kiểm tra được đánh dấu trong `configs/data.yaml`,
  docstring của script tương ứng và báo cáo.

## Dữ liệu, mã bên thứ ba và công cụ AI

- Dữ liệu UWB gốc: TIERS UWB Relative Localization Dataset, giấy phép MIT (xem `THIRD_PARTY.md`).
- Thư viện: PyTorch, NumPy, SciPy, scikit-learn, rosbags. Lớp GATv2, MAPPO, mô phỏng và bootstrap do nhóm tự cài.
- Mã nguồn được viết với sự hỗ trợ của trợ lý AI Claude (Anthropic); phạm vi sử dụng được công khai trong báo cáo.
