# E8 — đăng ký trước (viết 16/11/2026, TRƯỚC khi chạy đánh giá thật)

Mục đích: đo xem chất lượng bộ phát hiện (E3–E7) ảnh hưởng thế nào tới một nhiệm vụ phối hợp đa robot do
chính sách MARL điều khiển. Tài liệu này được commit lên GitHub trước khi chạy `e8_eval.py` trên Colab;
thời điểm commit là bằng chứng các giả thuyết và thiết kế được chốt trước.

## Môi trường (src/e8_env.py)
- 5 rover, vùng 24 × 24 m, 400 bước × 1 s, tốc độ 0,3 m/s; trôi định vị, mô hình lỗi (drift / spoof / spoof_one),
  xác suất lỗi 0,7, nhiễu UWB TIERS — giống hệt mô phỏng E0–E7.
- Nhiệm vụ phủ vùng: lưới 16 × 16 ô (1,5 m), 8 khu. Bản đồ "đã phủ" chung dựng từ vị trí tự khai được tin.
- Mỗi 20 bước, mỗi rover chọn khu (MARL); cấp thấp cố định: đi tới ô chưa phủ gần nhất trong khu.
- Rover bị gắn cờ: bỏ vị trí khai trong 100 bước trước và suốt lúc bị cờ (RETRO = 100, chốt trước).
- Bộ phát hiện chạy trực tuyến từ bước 30, báo động theo ngưỡng val của E3 (FPR 1%/bước).

## Chính sách (src/e8_train.py)
- MAPPO, tham số chung, critic tập trung, 300 vòng × 64 episode. Huấn luyện với nguồn tin cậy "noisy"
  (oracle + trễ 0–60 bước, bỏ sót 10%, báo nhầm 0–4%/mốc), không gắn với bộ phát hiện nào.
- 20 seed chính sách (10–29); chính sách seed s dùng bộ phát hiện seed s.
- Đánh giá: 300 episode cố định (seed 10^6 …), hành động lấy mẫu với bộ sinh số cố định (common random numbers).

## Điều kiện (cùng chính sách, cùng episode)
none · oracle · logistic · GRU · GRU+GAT · GRU+GAT + đổ oan nhân tạo cho hàng xóm trung thực của rover lỗi
(0,7%, 2,1%, 10% mỗi mốc) · heuristic + oracle · heuristic + GRU+GAT.

## Thước đo
- holes: tỉ lệ diện tích khai "đã phủ" nhưng thật ra chưa rover nào tới (an toàn nhiệm vụ).
- coverage: tỉ lệ diện tích rover thật sự đi qua (hiệu quả).
- Chính: trên episode có lỗi. Phụ: episode không lỗi, theo loại lỗi, tỉ lệ báo nhầm mỗi mốc.

## Giả thuyết
- H1: holes (episode có lỗi): none > logistic > GRU ≥ GRU+GAT ≥ oracle.
- H2: đổ oan làm mất công phủ lại → coverage giảm theo liều; holes không tăng.
  Dự đoán thêm: ở liều thực tế (0,7%) ảnh hưởng có thể quá nhỏ để phân biệt — nếu vậy sẽ báo đúng như thế.
- H3: MAPPO có coverage cao hơn heuristic (cùng nguồn tin cậy).

## Thống kê
- So sánh ghép cặp (cùng seed, cùng episode). KTC 95% bootstrap hai tầng (seed × episode), 2000 lần; kèm IQM qua seed.
- "Có ý nghĩa" chỉ khi KTC không chứa 0; ngược lại ghi "không phân biệt được ở mức 20 seed".

## Đã biết trước khi chạy (từ thử nghiệm sơ bộ với bộ phát hiện giả lập kém, 2 seed, 60 episode)
- holes và coverage đánh đổi nhau: càng ít tin (kể cả tin nhầm) thì holes càng giảm vì vùng bị bỏ được phủ lại.
  Vì vậy luôn báo CẢ HAI thước đo, không gộp thành một.
- Hành động lấy mẫu tốt hơn argmax (0,97 so với 0,94 coverage) → dùng lấy mẫu.
