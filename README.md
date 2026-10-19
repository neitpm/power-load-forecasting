# Electrical Power Load Forecasting (Dự báo phụ tải điện)

Dự án nghiên cứu và phát triển các mô hình học máy, học sâu tiên tiến (Kolmogorov-Arnold Networks - KAN, Graph Neural Networks - GNN, GRU/TCN) phục vụ bài toán **dự báo phụ tải điện** tại các trạm biến áp và lưới điện phân phối thuộc Điện lực Tây Ninh.

---

## 📌 Mục tiêu đề tài
1. **Dự báo phụ tải ngắn hạn (Short-term Load Forecasting - 30 phút):**
   - Dữ liệu thu thập chu kỳ 30 phút/lần từ hệ thống đo xa/SCADA.
   - Dự báo xu hướng (tăng/giảm) và giá trị công suất cụ thể ($P\text{ kW}$) của trạm biến áp lớn trong 30 phút tiếp theo dựa trên chuỗi thời gian lịch sử và tương quan với các xuất tuyến/điểm đo nhánh.
2. **Dự báo phụ tải trung & dài hạn (Monthly Load Forecasting):**
   - Phân tích và dự báo sản lượng điện năng tiêu thụ hàng tháng theo khách hàng và trạm công cộng giai đoạn 2019 – 2025 (7 năm).
3. **Ứng dụng kiến trúc mô hình mới:**
   - Khảo sát và ứng dụng **KAN (Kolmogorov-Arnold Networks)** / **KAN-Topology** kết hợp **GNN (Graph Neural Networks)** để mô hình hóa cấu trúc tô-pô lưới điện và quan hệ phi tuyến giữa các nút đo.

---

## 📂 Cấu trúc dự án

```text
power-load-forecasting/
├── code/
│   ├── batch_format_power_csvs.py     # Pipeline chuẩn hóa, lọc và chuyển đổi CSV sang Wide/Long format
│   ├── power_load_preprocessing_pipeline.py # Pipeline tiền xử lý & chia tập dữ liệu
│   ├── models_kan.py                  # Kiến trúc mạng Residual KAN
│   ├── train_baseline_models.py       # Huấn luyện và đánh giá các mô hình baseline
│   ├── kan_multi_seed_benchmark.py    # Thực nghiệm đa seed và ensemble cho KAN
│   └── generate_master_benchmark.py   # Tổng hợp bảng đối chuẩn hiệu năng
├── datasets/
│   ├── phu tai cac nam tinh theo thang/ # Dữ liệu điện năng tiêu thụ theo tháng (2019 - 2025)
│   ├── Tan Hung/                      # Dữ liệu đo đếm chi tiết 30 phút trạm Tân Hưng (2020 - 2025)
│   └── Suoi Ngo/                      # Dữ liệu đo đếm 30 phút trạm Suối Ngô (2024 - 2025)
├── document/                          # Tài liệu khảo sát, bài báo khoa học về KAN, GNN, Time Series
├── output_preprocessing/reports/      # Báo cáo đánh giá, đối chuẩn và tóm tắt tiền xử lý
├── Bao_Cao_Phu_Tai_Dien_KANTobolory.docx # Đề cương chi tiết luận văn
├── .gitignore                         # Danh sách tệp loại trừ của Git
└── README.md                          # Giới thiệu dự án
```

---

## 📊 Dữ liệu đo đếm (Datasets)
Dữ liệu đo đếm bao gồm các thông số kỹ thuật điện 3 pha thu thập theo chu kỳ 30 phút:
- $U_A, U_B, U_C$: Điện áp 3 pha (V)
- $I_A, I_B, I_C$: Dòng điện 3 pha (A)
- $\cos\varphi$: Hệ số công suất
- $P\text{ (kW)}$: Công suất tác dụng
- $Q\text{ (kVAr)}$: Công suất phản kháng

---

## 🚀 Hướng dẫn sử dụng

### Tiền xử lý dữ liệu đo đếm hàng loạt
Sử dụng script [batch_format_power_csvs.py](file:///code/batch_format_power_csvs.py) để chuẩn hóa định dạng thời gian và trải rộng dữ liệu (Wide format):

```bash
python code/batch_format_power_csvs.py --input_dir "datasets/Tan Hung/Tan Hung/224/2024" --out_dir "formatted_out" --freq 30min
```

---

## 📈 Benchmark & Evaluation Results (Master Comparison)

Experimental evaluation was conducted on an out-of-sample test set comprising **14,385 valid 30-minute records** partitioned across **16 continuous time segments** at **Feeder 480, Substation 224 Tan Hung (Tay Ninh Power Company)**. Evaluation interval: `2024-11-02 15:30:00` to `2025-08-30 23:30:00`.

### Benchmark Results Table

| Family | Model | MAE (kW) | RMSE (kW) | sMAPE (%) | $R^2$ Score | $\Delta\text{MAE vs Persistence}$ | $\Delta\text{RMSE vs Persistence}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 1. Persistence Naive ($P_{t}$) | 598.36 | 1160.44 | **13.35%** | 0.8770 | Baseline (0.00%) | Baseline (0.00%) |
| **Baseline** | 2. Seasonal Naive 24h ($\text{lag}_{48}$) | 1363.07 | 2495.50 | 30.41% | 0.4313 | -127.80% | -115.05% |
| **Baseline** | 3. Seasonal Naive 7d ($\text{lag}_{336}$) | 1882.99 | 3123.69 | 42.54% | 0.1090 | -214.69% | -169.18% |
| **Baseline** | 4. Moving Average (1.5h) | 1016.16 | 1586.70 | 21.74% | 0.7701 | -69.82% | -36.73% |
| **Classical ML** | 5. Ridge Regression | 730.96 | 1186.89 | 24.73% | 0.8714 | -22.16% | -2.28% |
| **Classical ML** | 6. XGBoost Regressor | 645.24 | 1155.21 | 23.30% | 0.8781 | -7.83% | +0.45% |
| **Classical ML** | 7. HistGradientBoosting Regressor | 648.01 | 1153.06 | 23.41% | 0.8786 | -8.30% | +0.64% |
| **Multi-seed Stability** | 8. Residual KAN Single (Mean $\pm$ Std)* | 528.99 $\pm$ 12.23 | 1033.35 $\pm$ 13.78 | 19.17% $\pm$ 0.32% | 0.9025 $\pm$ 0.0026 | +11.59% | +10.95% |
| **Proposed Model** | 9. **Residual KAN Ensemble (5 Seeds)\*\*** | **519.78** | **1019.95** | 18.96% | **0.9050** | **+13.13%** | **+12.11%** |

### Methodology & Data Leakage Control Notes

1. **Relative Improvement Formulation:**
   $$\Delta\text{Metric (\%)} = \frac{\text{Metric}_{\text{Persistence}} - \text{Metric}_{\text{Model}}}{\text{Metric}_{\text{Persistence}}} \times 100\%$$
   A positive value ($+$) indicates reduction in prediction error over the Persistence baseline.

2. **Data Leakage Control (Strict Out-of-Sample Separation):**
   - Feature scalers (`StandardScaler`) and imputers (`SimpleImputer`) were fitted **strictly on the training split** (03/2020 to 01/2024), and applied via transformation only to the validation and test splits.
   - Early stopping thresholds were monitored on the validation split (01/2024 to 11/2024). The test split remained completely untouched during hyperparameter tuning and model training.

3. `(*)` **Multi-seed Stability Assessment:**
   Evaluated across five independent random initializations (`seeds = [42, 101, 2024, 7, 999]`), confirming network convergence stability with low variance in MAE ($\sigma = 12.23\text{ kW}$) and RMSE ($\sigma = 13.78\text{ kW}$).

4. `(**)` **Ensemble Strategy:**
   Constructed via fixed uniform simple averaging ($\hat{y}_{\text{ens}} = \frac{1}{5}\sum_{i=1}^5 \hat{y}_i$), avoiding post-hoc weight tuning on test data to ensure zero data leakage.

5. **Key Takeaway & Scientific Rigor:**
   The Residual KAN ensemble reduced absolute and squared error ($\text{MAE} -13.13\%$, $\text{RMSE} -12.11\%$, $R^2 = 0.9050$), but did not dominate the persistence baseline on every metric ($\text{Persistence Naive}$ remained superior on sMAPE at $13.35\%$). This result highlights the critical importance of evaluating short-horizon load forecasting models against strong naive baselines.
