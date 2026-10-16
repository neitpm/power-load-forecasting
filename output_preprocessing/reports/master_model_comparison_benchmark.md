# BẢNG TỔNG HỢP ĐỐI CHIẾU MÔ HÌNH DỰ BÁO PHỤ TẢI (TRẠM 224 TÂN HƯNG)

| Phân nhóm | Mô hình | MAE (kW) | RMSE (kW) | MAE (MW) | RMSE (MW) | sMAPE (%) | R2 Score | ΔMAE vs Persistence (%) | ΔRMSE vs Persistence (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline / ML Truyền thống | 1. Persistence Naive (P_t) | 598.36 | 1160.44 | 0.5984 | 1.1604 | 13.35% | 0.8770 | Baseline (0.00%) | Baseline (0.00%) |
| Baseline / ML Truyền thống | 2. Seasonal Naive 24h (lag_48) | 1363.07 | 2495.50 | 1.3631 | 2.4955 | 30.41% | 0.4313 | -127.80% | -115.05% |
| Baseline / ML Truyền thống | 3. Seasonal Naive 7d (lag_336) | 1882.99 | 3123.69 | 1.8830 | 3.1237 | 42.54% | 0.1090 | -214.69% | -169.18% |
| Baseline / ML Truyền thống | 4. Moving Average (1.5h) | 1016.16 | 1586.70 | 1.0162 | 1.5867 | 21.74% | 0.7701 | -69.82% | -36.73% |
| Baseline / ML Truyền thống | 5. Ridge Regression | 730.96 | 1186.89 | 0.7310 | 1.1869 | 24.73% | 0.8714 | -22.16% | -2.28% |
| Baseline / ML Truyền thống | 6. XGBoost Regressor | 645.24 | 1155.21 | 0.6452 | 1.1552 | 23.30% | 0.8781 | -7.83% | +0.45% |
| Baseline / ML Truyền thống | 7. HistGradientBoosting Regressor | 648.01 | 1153.06 | 0.6480 | 1.1531 | 23.41% | 0.8786 | -8.30% | +0.64% |
| Thống kê độ ổn định (5 Seed) | 8. Residual KAN đơn lẻ (Mean ± Std qua 5 Seed)* | 528.99 ± 12.23 | 1033.35 ± 13.78 | 0.5290 | 1.0333 | 19.17% ± 0.32% | 0.9025 ± 0.0026 | +11.59% | +10.95% |
| Mô hình đề xuất (Deep Learning) | 9. Residual KAN Ensemble (Trung bình đều 5 Seed)** | 519.78 | 1019.95 | 0.5198 | 1.0199 | 18.96% | 0.9050 | +13.13% | +12.11% |

### Ghi chú phương pháp và phạm vi áp dụng (Dành cho Luận văn Thạc sĩ):

1. **Phạm vi dữ liệu thực nghiệm:**
   - Tập kiểm thử gồm **14.385 mẫu hợp lệ**, được phân thành **16 phân đoạn thời gian liên tục**; quy trình bootstrap kiểm định không vượt qua các khoảng đứt gãy dữ liệu.
   - Khoảng thời gian kiểm thử chính xác: **02/11/2024 15:30:00 đến 30/08/2025 23:30:00**.
   - Đối tượng kiểm định: **Lộ 480 Trạm biến áp 224 Tân Hưng** (Công ty Điện lực Tây Ninh).
   - *Lưu ý học thuật:* Kết quả phản ánh hiệu năng trên tập kiểm định cụ thể này; việc mở rộng sang các xuất tuyến khác (như trạm Suối Ngô hoặc các năm khác) cần có thực nghiệm tái thẩm định riêng biệt.

2. **Công thức tính tỷ lệ cải thiện:**
   $$\text{Improvement (\%)} = \frac{\text{Metric}_{\text{Persistence}} - \text{Metric}_{\text{Model}}}{\text{Metric}_{\text{Persistence}}} \times 100\%$$
   - Giá trị **dương (+)**: Mô hình giảm sai số (tốt hơn Persistence).
   - Giá trị **âm (-)**: Mô hình có sai số lớn hơn Persistence (kém hơn).

3. **Phân biệt hai dòng kết quả KAN:**
   - `(*)` **Dòng 8 (Mean ± Std):** Thống kê độ ổn định của mô hình đơn qua 5 lần khởi tạo ngẫu nhiên khác nhau (`seeds = [42, 101, 2024, 7, 999]`), thể hiện tính hội tụ vững chắc của mạng.
   - `(**)` **Dòng 9 (Ensemble):** Mô hình kết hợp bằng **trung bình đều cố định** $(\hat{y}_{\text{ens}} = \frac{1}{5}\sum_{i=1}^5 \hat{y}_i)$, tuyệt đối không tối ưu hóa trọng số trên tập kiểm thử nhằm đảm bảo khách quan (Zero Data Leakage).

4. **Đánh giá toàn diện:**
   - **Residual KAN** đạt ưu thế rõ rệt và vượt trội về **MAE, RMSE và R²**.
   - **Persistence Naive** vẫn giữ vị trí dẫn đầu về **sMAPE (13.35%)** và là mốc baseline bắt buộc phải có để đối chiếu trong bài toán dự báo phụ tải ngắn hạn 30 phút.
