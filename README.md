# Phát hiện robot báo sai vị trí bằng mạng nơ-ron đồ thị theo thời gian

Bài tập lớn môn Học máy nâng cao. Bài toán: trong một đội rover không có GPS, mỗi rover tự khai vị trí
và đo khoảng cách UWB tới đồng đội. Mô hình phân loại từng rover tại mỗi bước:
trung thực / trôi định vị (F-drift) / khai giả (A-spoof). Hướng A + C (GRU + GAT).

## Cấu trúc
```
configs/data.yaml      các tập dữ liệu và khoảng seed riêng của từng tập
configs/seeds.yaml     seed dò siêu tham số và seed báo cáo
src/noise_tiers.py     trích mô hình nhiễu UWB từ dữ liệu thật TIERS -> data/noise_tiers.npz
src/sim.py             mô phỏng đội rover, định vị tự khai, đo UWB, các loại lỗi
src/make_dataset.py    sinh các tập dữ liệu -> data/<split>.npz
src/check_data.py      E0: kiểm tra phân bố phần dư theo nhóm
src/features.py        phần dư range và 12 đặc trưng cửa sổ (nhân quả)
src/metrics.py         thước đo chung: macro-F1, recall khi báo nhầm <= 1%, độ trễ phát hiện, ECE
src/baselines.py       E1: baseline ngưỡng và hồi quy logistic
src/checksum.py        mã băm dữ liệu để đối chiếu giữa các máy
src/runlog.py          ghi nhật ký mỗi lần chạy
results/runs_pro/      kết quả và runlog.csv của tài khoản Colab Pro
results/runs_free/     kết quả và runlog.csv của tài khoản Colab miễn phí
```

## Tái hiện
```bash
pip install -r requirements.txt
python src/make_dataset.py --split all        # khoảng 2–3 phút trên CPU
python src/check_data.py data/val.npz
python src/checksum.py data/train.npz data/val.npz data/test.npz
```
`data/noise_tiers.npz` đã có sẵn trong repo. Để tạo lại từ dữ liệu gốc:
```bash
git clone --depth 1 https://github.com/TIERS/uwb-relative-localization-dataset ../tiers
pip install rosbags==0.11.9
python src/noise_tiers.py --tiers ../tiers --out data/noise_tiers.npz
```

## Quy tắc thực nghiệm
- Mỗi tập dữ liệu dùng khoảng seed riêng (xem `configs/data.yaml`), chia theo episode.
- Siêu tham số chỉ dò trên tập `val`. Tập `test` và các tập `ood_*` chỉ chạy một lần sau khi chốt cấu hình.
- Mô hình không được dùng `p_true` hay `label` làm đặc trưng; `p_true` chỉ để phân tích.
- Mỗi lần chạy ghi một dòng bằng `src/runlog.py` vào `results/runs_<tài khoản>/runlog.csv`.

## Dữ liệu bên thứ ba
Xem `THIRD_PARTY.md`.
