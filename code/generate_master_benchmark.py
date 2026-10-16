# -*- coding: utf-8 -*-
"""
=============================================================================
BẢNG TỔNG HỢP SO SÁNH TOÀN DIỆN CÁC MÔ HÌNH DỰ BÁO PHỤ TẢI ĐIỆN 30 PHÚT
(PHIÊN BẢN CHUẨN MỰC HỌC THUẬT CHO LUẬN VĂN THẠC SĨ)
=============================================================================
Đặc điểm hiệu chỉnh học thuật:
1. Tách bạch rõ ràng 2 cách thể hiện của KAN:
   - Dòng 8: Đánh giá độ ổn định của mô hình đơn qua 5 seed (Mean ± Std).
   - Dòng 9: Mô hình kết hợp độc lập (5-Seed Ensemble) - một mô hình dự báo riêng biệt.
2. Công thức cải thiện (Improvement) được định nghĩa nhất quán:
   Improvement (%) = (Metric_Baseline - Metric_Model) / Metric_Baseline * 100%
   (Giá trị dương: mô hình cải thiện/giảm sai số; Giá trị âm: mô hình kém hơn Persistence).
3. Ghi rõ phạm vi nghiên cứu và giới hạn thực nghiệm:
   - Áp dụng trên tập kiểm thử độc lập (14,385 mẫu, 11/2024 -> 08/2025).
   - Đối tượng kiểm chứng: Lộ 480 Trạm 224 Tân Hưng (Điện lực Tây Ninh).
=============================================================================
"""

import os
import sys
import json
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np


def generate_master_benchmark_table():
    reports_dir = Path("output_preprocessing/reports")

    # 1. Đọc kết quả baseline
    df_base = pd.read_csv(reports_dir / "baseline_models_comparison.csv")

    # 2. Đọc kết quả KAN 5 seed
    with open(reports_dir / "kan_multi_seed_stability.json", "r", encoding="utf-8") as f:
        kan_data = json.load(f)

    stats = kan_data["summary_statistics"]
    ens = kan_data["ensemble_5_seeds"]["metrics"]

    master_rows = []

    for _, row in df_base.iterrows():
        master_rows.append({
            "Phân nhóm": "Baseline / ML Truyền thống",
            "Mô hình": row["model_name"],
            "MAE (kW)": f"{row['MAE_kW']:.2f}",
            "RMSE (kW)": f"{row['RMSE_kW']:.2f}",
            "MAE (MW)": f"{row['MAE_MW']:.4f}",
            "RMSE (MW)": f"{row['RMSE_MW']:.4f}",
            "sMAPE (%)": f"{row['sMAPE_pct']:.2f}%",
            "R2 Score": f"{row['R2_score']:.4f}",
            "ΔMAE vs Persistence (%)": f"{((598.36 - row['MAE_kW']) / 598.36 * 100):+.2f}%" if "Persistence" not in row["model_name"] else "Baseline (0.00%)",
            "ΔRMSE vs Persistence (%)": f"{((1160.44 - row['RMSE_kW']) / 1160.44 * 100):+.2f}%" if "Persistence" not in row["model_name"] else "Baseline (0.00%)",
        })

    # Dòng 8: Đánh giá độ ổn định của mô hình đơn KAN qua 5 seed (Mean ± Std)
    master_rows.append({
        "Phân nhóm": "Thống kê độ ổn định (5 Seed)",
        "Mô hình": "8. Residual KAN đơn lẻ (Mean ± Std qua 5 Seed)*",
        "MAE (kW)": f"{stats['MAE_kW_mean']:.2f} ± {stats['MAE_kW_std']:.2f}",
        "RMSE (kW)": f"{stats['RMSE_kW_mean']:.2f} ± {stats['RMSE_kW_std']:.2f}",
        "MAE (MW)": f"{stats['MAE_kW_mean']/1000:.4f}",
        "RMSE (MW)": f"{stats['RMSE_kW_mean']/1000:.4f}",
        "sMAPE (%)": f"{stats['sMAPE_pct_mean']:.2f}% ± {stats['sMAPE_pct_std']:.2f}%",
        "R2 Score": f"{stats['R2_score_mean']:.4f} ± {stats['R2_score_std']:.4f}",
        "ΔMAE vs Persistence (%)": f"+{stats['Improvement_MAE_pct_mean']:.2f}%",
        "ΔRMSE vs Persistence (%)": f"+{stats['Improvement_RMSE_pct_mean']:.2f}%"
    })

    # Dòng 9: Mô hình kết hợp độc lập (5-Seed Ensemble)
    master_rows.append({
        "Phân nhóm": "Mô hình đề xuất (Deep Learning)",
        "Mô hình": "9. Residual KAN Ensemble (Trung bình 5 Seed)**",
        "MAE (kW)": f"{ens['MAE_kW']:.2f}",
        "RMSE (kW)": f"{ens['RMSE_kW']:.2f}",
        "MAE (MW)": f"{ens['MAE_MW']:.4f}",
        "RMSE (MW)": f"{ens['RMSE_MW']:.4f}",
        "sMAPE (%)": f"{ens['sMAPE_pct']:.2f}%",
        "R2 Score": f"{ens['R2_score']:.4f}",
        "ΔMAE vs Persistence (%)": f"+{ens['Improvement_MAE_pct']:.2f}%",
        "ΔRMSE vs Persistence (%)": f"+{ens['Improvement_RMSE_pct']:.2f}%"
    })

    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(reports_dir / "master_model_comparison_benchmark.csv", index=False, encoding="utf-8-sig")

    # Xuất định dạng Markdown chuẩn kèm chú thích học thuật chặt chẽ
    headers = list(df_master.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, r in df_master.iterrows():
        lines.append("| " + " | ".join([str(r[c]) for c in headers]) + " |")
    md_table = "\n".join(lines)

    notes = """
### Ghi chú phương pháp và phạm vi áp dụng (Dành cho Luận văn Thạc sĩ):

1. **Phạm vi dữ liệu thực nghiệm:**
   - Tập kiểm thử gồm **14.385 mẫu đo đồng nhất** (bước nhảy 30 phút liên tục không rò rỉ dữ liệu).
   - Giai đoạn kiểm thử cố định: **01/11/2024 đến 30/08/2025**.
   - Đối tượng kiểm định: **Lộ 480 Trạm biến áp 224 Tân Hưng** (Công ty Điện lực Tây Ninh).
   - *Lưu ý học thuật:* Kết quả phản ánh hiệu năng trên tập kiểm định cụ thể này; việc mở rộng sang các xuất tuyến khác (như trạm Suối Ngô hoặc các năm khác) cần có thực nghiệm tái thẩm định riêng biệt.

2. **Công thức tính tỷ lệ cải thiện:**
   $$\\text{Improvement (\\%)} = \\frac{\\text{Metric}_{\\text{Persistence}} - \\text{Metric}_{\\text{Model}}}{\\text{Metric}_{\\text{Persistence}}} \\times 100\\%$$
   - Giá trị **dương (+)**: Mô hình giảm sai số (tốt hơn Persistence).
   - Giá trị **âm (-)**: Mô hình có sai số lớn hơn Persistence (kém hơn).

3. **Phân biệt hai dòng kết quả KAN:**
   - `(*)` **Dòng 8 (Mean ± Std):** Thống kê độ ổn định của mô hình đơn qua 5 lần khởi tạo ngẫu nhiên khác nhau (Random Seeds: 42, 101, 2024, 7, 999), thể hiện tính hội tụ vững chắc của mạng.
   - `(**)` **Dòng 9 (Ensemble):** Một mô hình dự báo kết hợp độc lập bằng cách lấy trung bình đầu ra của 5 mạng, đạt mức sai số thấp nhất trên tập kiểm thử.

4. **Đánh giá toàn diện:**
   - **Residual KAN** đạt ưu thế rõ rệt và vượt trội về **MAE, RMSE và R²**.
   - **Persistence Naive** vẫn giữ vị trí dẫn đầu về **sMAPE (13.35%)** và là mốc baseline bắt buộc phải có để đối chiếu trong bài toán dự báo phụ tải ngắn hạn 30 phút.
"""

    with open(reports_dir / "master_model_comparison_benchmark.md", "w", encoding="utf-8") as f:
        f.write("# BẢNG TỔNG HỢP ĐỐI CHIẾU MÔ HÌNH DỰ BÁO PHỤ TẢI (TRẠM 224 TÂN HƯNG)\n\n")
        f.write(md_table)
        f.write("\n\n" + notes.strip() + "\n")

    print("✅ Đã cập nhật bảng tổng hợp chuẩn mực học thuật tại:")
    print(f"   - {reports_dir / 'master_model_comparison_benchmark.csv'}")
    print(f"   - {reports_dir / 'master_model_comparison_benchmark.md'}")


if __name__ == "__main__":
    generate_master_benchmark_table()
