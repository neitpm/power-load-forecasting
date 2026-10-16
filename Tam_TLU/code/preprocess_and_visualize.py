# -*- coding: utf-8 -*-
"""
Module: preprocess_and_visualize.py
Chức năng:
1. Tiền xử lý dữ liệu phụ tải (Clean anomalies, Impute missing values, Resample to 30-min interval).
2. Tự động sinh báo cáo kiểm định chất lượng gồm 5 biểu đồ chuẩn EDA & Validation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def preprocess_power_data(file_path):
    """
    Đọc và tiền xử lý dữ liệu thô:
    - Chuẩn hóa TIME, ép kiểu số.
    - Lọc bỏ các điểm đo lỗi điện áp vọt cao (>20kV) hoặc công suất âm phi thực tế.
    - Nội suy tuyến tính dữ liệu khuyết.
    - Làm tròn và gom nhóm theo mốc chu kỳ 30 phút.
    """
    df = pd.read_csv(file_path)
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
    df = df.dropna(subset=["TIME"]).sort_values("TIME").reset_index(drop=True)
    
    num_cols = ["UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    # Bản sao để lưu giữ dữ liệu thô phục vụ đối chiếu trước/sau làm sạch
    df_raw = df.copy()
    
    # Tiền xử lý lọc nhiễu vật lý
    # Lọc áp vượt định mức (quy về NaN để nội suy sau)
    for col in ["UA", "UB", "UC"]:
        df.loc[df[col] > 20000, col] = np.nan
        
    # Lọc P_kW âm hoặc quá cao đột biến
    df.loc[df["P_kW"] < 0, "P_kW"] = np.nan
    df.loc[df["P_kW"] > 20000, "P_kW"] = np.nan
    
    # Điền khuyết bằng phương pháp nội suy tuyến tính (Linear Interpolation)
    df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')
    
    # Chuyển đổi tần suất dữ liệu về đều đặn mỗi 30 phút cho từng trạm
    cleaned_dfs = []
    for node, group in df.groupby("TenDiemDo"):
        group = group.set_index("TIME")
        # Gom nhóm 30 phút và lấy giá trị trung bình
        group_resampled = group[num_cols].resample("30Min").mean().interpolate(method='linear')
        group_resampled["TenDiemDo"] = node
        group_resampled = group_resampled.reset_index()
        cleaned_dfs.append(group_resampled)
        
    df_clean = pd.concat(cleaned_dfs, ignore_index=True)
    return df_raw, df_clean

def generate_preprocessing_report(df_raw, df_clean, output_dir):
    """
    Tự động sinh và lưu 5 biểu đồ kiểm định chất lượng dữ liệu
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # -----------------------------------------------------
    # 1. Biểu đồ kiểm tra làm sạch & điền khuyết (Before vs. After)
    # -----------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    sample_node = df_clean["TenDiemDo"].unique()[0]
    
    raw_sub = df_raw[df_raw["TenDiemDo"] == sample_node].set_index("TIME").sort_index()
    clean_sub = df_clean[df_clean["TenDiemDo"] == sample_node].set_index("TIME").sort_index()
    
    # Chọn dải thời gian 1 tuần để quan sát rõ
    start_date = clean_sub.index.min()
    end_date = start_date + pd.Timedelta(days=7)
    
    ax1.plot(raw_sub.loc[start_date:end_date].index, raw_sub.loc[start_date:end_date, "P_kW"], 
             'r.', label='Dữ liệu thô (Bất thường / Khuyết)', alpha=0.6, markersize=8)
    ax1.plot(clean_sub.loc[start_date:end_date].index, clean_sub.loc[start_date:end_date, "P_kW"], 
             'b-', label='Dữ liệu sau khi xử lý (30-min Resampled)', linewidth=1.5)
    
    ax1.set_title(f"So sánh trước/sau tiền xử lý - Điểm đo {sample_node} (Thời gian: 1 Tuần)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Thời gian")
    ax1.set_ylabel("Công suất P (kW)")
    ax1.legend()
    fig1.savefig(output_path / "01_before_vs_after_imputation.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # -----------------------------------------------------
    # 2. Biểu đồ phân phối công suất & xử lý ngoại lai
    # -----------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram phân phối P_kW thô và sạch
    sns.histplot(df_raw["P_kW"].dropna(), bins=50, kde=True, ax=axes2[0], color='red', label='Thô (Gồm lỗi)', alpha=0.5)
    sns.histplot(df_clean["P_kW"], bins=50, kde=True, ax=axes2[0], color='blue', label='Sạch', alpha=0.5)
    axes2[0].set_title("Phân phối tần suất P_kW", fontsize=12, fontweight='bold')
    axes2[0].set_xlabel("Công suất P (kW)")
    axes2[0].legend()
    
    # Boxplot so sánh
    plot_data = pd.DataFrame({
        'P_kW': pd.concat([df_raw["P_kW"].dropna(), df_clean["P_kW"]]),
        'Trạng thái': ['Thô']*len(df_raw["P_kW"].dropna()) + ['Sau xử lý']*len(df_clean["P_kW"])
    })
    sns.boxplot(x='Trạng thái', y='P_kW', data=plot_data, ax=axes2[1], palette=['#ff4d4d', '#1a75ff'])
    axes2[1].set_title("Biểu đồ hộp Boxplot của P_kW", fontsize=12, fontweight='bold')
    axes2[1].set_ylabel("Công suất P (kW)")
    
    fig2.suptitle("Phân phối và giới hạn ngoại lai của Công suất P_kW", fontsize=14, fontweight='bold')
    fig2.savefig(output_path / "02_distribution_and_outliers.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # -----------------------------------------------------
    # 3. Biểu đồ đường cong phụ tải trạm lớn & các xuất tuyến
    # -----------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    
    # Pivot dữ liệu để lấy chuỗi P_kW của từng xuất tuyến nhánh
    pivot_clean = df_clean.pivot(index='TIME', columns='TenDiemDo', values='P_kW').fillna(0)
    # Công suất tổng
    pivot_clean['Tổng trạm Suối Ngô'] = pivot_clean.sum(axis=1)
    
    sample_range = pivot_clean.iloc[:336] # Chọn mẫu khoảng 1 tuần (336 mốc 30-phút)
    
    for col in sample_range.columns:
        if col == 'Tổng trạm Suối Ngô':
            ax3.plot(sample_range.index, sample_range[col], label=col, linewidth=2.5, color='black', linestyle='--')
        else:
            ax3.plot(sample_range.index, sample_range[col], label=col, alpha=0.7, linewidth=1)
            
    ax3.set_title("Đường cong phụ tải Tổng trạm & Các xuất tuyến lộ nhánh con", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Thời gian")
    ax3.set_ylabel("Công suất P (kW)")
    ax3.legend(loc='upper right')
    fig3.savefig(output_path / "03_load_profile_branches.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # -----------------------------------------------------
    # 4. Biểu đồ kiểm tra biến mục tiêu dự báo
    # -----------------------------------------------------
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tính delta P
    df_diffs = []
    for node, gp in df_clean.groupby("TenDiemDo"):
        gp = gp.sort_values("TIME")
        gp["Delta_P"] = gp["P_kW"].diff()
        df_diffs.append(gp.dropna(subset=["Delta_P"]))
    df_target = pd.concat(df_diffs, ignore_index=True)
    
    # Phân phối Delta P
    sns.histplot(df_target["Delta_P"], bins=100, kde=True, ax=axes4[0], color='purple')
    axes4[0].set_title("Phân phối độ biến thiên Delta_P (P[t+1] - P[t])", fontsize=12, fontweight='bold')
    axes4[0].set_xlabel("Delta P (kW)")
    
    # Phân loại xu hướng: Tăng (Delta_P > threshold), Giảm (<-threshold), Ổn định
    threshold = 1.0 # Ngưỡng biến thiên kW để phân loại xu hướng
    conditions = [
        (df_target["Delta_P"] > threshold),
        (df_target["Delta_P"] < -threshold)
    ]
    choices = ['TĂNG', 'GIẢM']
    df_target["Xu_Huong"] = np.select(conditions, choices, default='ỔN ĐỊNH')
    
    counts = df_target["Xu_Huong"].value_counts(normalize=True) * 100
    counts.plot(kind='bar', ax=axes4[1], color=['grey', 'green', 'red'])
    axes4[1].set_title("Tỷ lệ nhãn xu hướng biến thiên công suất", fontsize=12, fontweight='bold')
    axes4[1].set_ylabel("Tỷ lệ (%)")
    for i, v in enumerate(counts):
        axes4[1].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
        
    fig4.suptitle("Kiểm định biến mục tiêu dự báo (Target Verification)", fontsize=14, fontweight='bold')
    fig4.savefig(output_path / "04_target_verification.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)

    # -----------------------------------------------------
    # 5. Biểu đồ nhiệt phụ tải theo chu kỳ (Weekday x Hour Heatmap)
    # -----------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    
    df_heatmap = df_clean.copy()
    df_heatmap["Hour"] = df_heatmap["TIME"].dt.hour + df_heatmap["TIME"].dt.minute/60
    df_heatmap["DayOfWeek"] = df_heatmap["TIME"].dt.day_name()
    
    # Sắp xếp thứ trong tuần một cách chuẩn xác
    weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Pivot dữ liệu lấy trung bình công suất
    pivot_heatmap = df_heatmap.groupby(['DayOfWeek', 'Hour'])['P_kW'].mean().unstack(level=-1)
    pivot_heatmap = pivot_heatmap.reindex(weekdays_order)
    
    sns.heatmap(pivot_heatmap, cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Công suất trung bình (kW)'})
    ax5.set_title("Biểu đồ nhiệt phụ tải theo chu kỳ (Thứ trong tuần x Giờ trong ngày)", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Giờ trong ngày")
    ax5.set_ylabel("Thứ trong tuần")
    
    fig5.savefig(output_path / "05_load_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    print(f"Thành công! Đã sinh và lưu 5 biểu đồ chất lượng tại: {output_path.resolve()}")

if __name__ == '__main__':
    # Demo chạy thử với dữ liệu sẵn có của Suối Ngô
    input_file = '/content/drive/MyDrive/power-load-forecasting/Tam_TLU/datasets/Suoi Ngo/Suoi Ngo/2025/output_suoingo/SUOINGO_2025_T1_to_T9_merged.csv'
    output_charts_dir = '/content/drive/MyDrive/power-load-forecasting/Tam_TLU/datasets/Suoi Ngo/Suoi Ngo/2025/output_suoingo/reports/'
    
    print("Bắt đầu quy trình xử lý dữ liệu...")
    raw, clean = preprocess_power_data(input_file)
    
    print("Bắt đầu vẽ biểu đồ báo cáo...")
    generate_preprocessing_report(raw, clean, output_charts_dir)
