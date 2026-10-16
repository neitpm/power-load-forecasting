# -*- coding: utf-8 -*-
"""
=============================================================================
BẢNG TỔNG HỢP SO SÁNH TOÀN DIỆN CÁC MÔ HÌNH DỰ BÁO PHỤ TẢI ĐIỆN 30 PHÚT
(TỪ BASELINE ĐẾN MÔ HÌNH HỌC MÁY GBDT VÀ MẠNG NƠ-RON DEEP LEARNING KAN)
=============================================================================
Đánh giá trên CHÍNH XÁC cùng một tập con 14,385 mẫu Test đồng nhất (Zero Leakage).
Đơn vị kép: kW và MW.
Bao gồm:
1. Persistence Naive (Baseline cơ sở)
2. Seasonal Naive 24h
3. Seasonal Naive 7d
4. Moving Average (1.5h)
5. Ridge Regression
6. XGBoost Regressor
7. HistGradientBoosting Regressor
8. Residual KAN (5-Seed Average: Mean ± Std)
9. Residual KAN (5-Seed Ensemble Mean)
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
            "model_family": "Baseline / ML Truyền thống",
            "model_name": row["model_name"],
            "MAE_kW": f"{row['MAE_kW']:.2f}",
            "RMSE_kW": f"{row['RMSE_kW']:.2f}",
            "MAE_MW": f"{row['MAE_MW']:.4f}",
            "RMSE_MW": f"{row['RMSE_MW']:.4f}",
            "sMAPE_pct": f"{row['sMAPE_pct']:.2f}%",
            "R2_score": f"{row['R2_score']:.4f}",
            "improvement_MAE_vs_pers": f"{((598.36 - row['MAE_kW']) / 598.36 * 100):+.2f}%" if "Persistence" not in row["model_name"] else "Baseline",
            "improvement_RMSE_vs_pers": f"{((1160.44 - row['RMSE_kW']) / 1160.44 * 100):+.2f}%" if "Persistence" not in row["model_name"] else "Baseline",
        })
        
    # Thêm dòng KAN (Mean ± Std)
    master_rows.append({
        "model_family": "Mô hình Đề xuất (Deep Learning KAN)",
        "model_name": "8. Residual KAN (5-Seed Mean ± Std)",
        "MAE_kW": f"{stats['MAE_kW_mean']:.2f} ± {stats['MAE_kW_std']:.2f}",
        "RMSE_kW": f"{stats['RMSE_kW_mean']:.2f} ± {stats['RMSE_kW_std']:.2f}",
        "MAE_MW": f"{stats['MAE_kW_mean']/1000:.4f}",
        "RMSE_MW": f"{stats['RMSE_kW_mean']/1000:.4f}",
        "sMAPE_pct": f"{stats['sMAPE_pct_mean']:.2f}% ± {stats['sMAPE_pct_std']:.2f}%",
        "R2_score": f"{stats['R2_score_mean']:.4f} ± {stats['R2_score_std']:.4f}",
        "improvement_MAE_vs_pers": f"+{stats['Improvement_MAE_pct_mean']:.2f}%",
        "improvement_RMSE_vs_pers": f"+{stats['Improvement_RMSE_pct_mean']:.2f}%"
    })
    
    # Thêm dòng KAN Ensemble
    master_rows.append({
        "model_family": "Mô hình Đề xuất (Deep Learning KAN)",
        "model_name": "9. Residual KAN (5-Seed Ensemble)",
        "MAE_kW": f"{ens['MAE_kW']:.2f}",
        "RMSE_kW": f"{ens['RMSE_kW']:.2f}",
        "MAE_MW": f"{ens['MAE_MW']:.4f}",
        "RMSE_MW": f"{ens['RMSE_MW']:.4f}",
        "sMAPE_pct": f"{ens['sMAPE_pct']:.2f}%",
        "R2_score": f"{ens['R2_score']:.4f}",
        "improvement_MAE_vs_pers": f"+{ens['Improvement_MAE_pct']:.2f}%",
        "improvement_RMSE_vs_pers": f"+{ens['Improvement_RMSE_pct']:.2f}%"
    })
    
    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(reports_dir / "master_model_comparison_benchmark.csv", index=False, encoding="utf-8-sig")
    
    # Xuất file Markdown để dễ nhúng vào báo cáo/luận văn
    headers = list(df_master.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, r in df_master.iterrows():
        lines.append("| " + " | ".join([str(r[c]) for c in headers]) + " |")
    md_table = "\n".join(lines)
    
    with open(reports_dir / "master_model_comparison_benchmark.md", "w", encoding="utf-8") as f:
        f.write("# BẢNG TỔNG HỢP ĐỐI CHIẾU MÔ HÌNH TOÀN DIỆN (14,385 MẪU TEST TÂN HƯNG)\n\n")
        f.write(md_table)
        f.write("\n\n*Ghi chú:* Toàn bộ mô hình được đánh giá trên cùng tập mẫu Test độc lập không rò rỉ dữ liệu.\n")
        
    print("✅ Đã tạo bảng tổng hợp hoàn chỉnh tại:")
    print(f"   - {reports_dir / 'master_model_comparison_benchmark.csv'}")
    print(f"   - {reports_dir / 'master_model_comparison_benchmark.md'}")

if __name__ == "__main__":
    generate_master_benchmark_table()
