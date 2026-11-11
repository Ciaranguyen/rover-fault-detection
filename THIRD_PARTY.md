# Dữ liệu và mã nguồn bên thứ ba

## TIERS UWB Relative Localization Dataset
- Nguồn: https://github.com/TIERS/uwb-relative-localization-dataset
- Giấy phép: MIT License, Copyright (c) 2023 TIERS
- Bài báo: P. Torrico Morón, S. Salimpour, L. Fu, X. Yu, J. Peña Queralta, T. Westerlund.
  "Benchmarking UWB-Based Infrastructure-Free Positioning and Multi-Robot Relative Localization:
  Dataset and Characterization." arXiv:2305.08532, 2023.
- Cách dùng trong dự án: `src/noise_tiers.py` đọc rosbag case 3 và case 4, so khoảng cách UWB với
  khoảng cách thật từ OptiTrack, lưu độ lệch từng cặp và phần dư vào `data/noise_tiers.npz`.
  Dữ liệu gốc không được sao chép vào repo này.
