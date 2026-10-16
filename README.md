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
│   └── batch_format_power_csvs.py     # Pipeline chuẩn hóa, lọc và chuyển đổi CSV sang Wide/Long format
├── datasets/
│   ├── phu tai cac nam tinh theo thang/ # Dữ liệu điện năng tiêu thụ theo tháng (2019 - 2025)
│   ├── Tan Hung/                      # Dữ liệu đo đếm chi tiết 30 phút trạm Tân Hưng (2020 - 2025)
│   └── Suoi Ngo/                      # Dữ liệu đo đếm 30 phút trạm Suối Ngô (2024 - 2025)
├── document/                          # Tài liệu khảo sát, bài báo khoa học về KAN, GNN, Time Series
├── Bao_Cao_Phu_Tai_Dien_KANTobolory.docx # Đề cương chi tiết luận văn
├── porm.txt                           # Ghi chú bài toán dự báo ngắn hạn 30 phút
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
