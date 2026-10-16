# -*- coding: utf-8 -*-
"""
=============================================================================
PIPELINE TIỀN XỬ LÝ DỮ LIỆU DỰ BÁO PHỤ TẢI ĐIỆN 30 PHÚT (POWER LOAD FORECASTING)
=============================================================================
Dự án: Nghiên cứu mô hình học sâu dự báo phụ tải điện trạm Điện lực Tây Ninh
Mục tiêu: Xây dựng pipeline tiền xử lý chuẩn mực, không rò rỉ dữ liệu (No Data Leakage),
         tự động tạo đặc trưng và xuất tập dữ liệu huấn luyện cho bài toán dự báo 30 phút.
=============================================================================
"""

import os
import sys
import glob
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập hiển thị đồ thị
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

NUMERIC_COLS = ["UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr"]


# =============================================================================
# 1. TÌM KIẾM VÀ ĐỌC FILE CSV
# =============================================================================
def find_csv_files(data_dir: Path) -> list[Path]:
    """
    Quét đệ quy tìm toàn bộ file CSV trong thư mục.
    Bỏ qua các file nén zip/7z và các file output/báo cáo đã tạo trước đó.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {data_dir}")

    all_csvs = sorted(data_dir.rglob("*.csv"))
    valid_csvs = []
    
    # Danh sách các từ khóa cần loại trừ (file kết quả, báo cáo tạm)
    exclude_keywords = ["report", "summary", "cleaned", "featured", "supervised", "train", "validation", "test", "merged_with_features"]

    for p in all_csvs:
        # Bỏ qua nếu tên file chứa từ khóa loại trừ
        if any(kw in p.name.lower() for kw in exclude_keywords):
            continue
        valid_csvs.append(p)

    return valid_csvs


# =============================================================================
# 2. CHUẨN HÓA THỜI GIAN VÀ CỘT DỮ LIỆU
# =============================================================================
def parse_time_robust(df: pd.DataFrame) -> pd.Series:
    """
    Hỗ trợ cả 2 trường hợp:
    - TIME đã chứa ngày giờ đầy đủ (ví dụ: '2024-01-01 00:00:04.000')
    - Ngày và giờ nằm ở 2 cột riêng (ví dụ: DATE + TIME)
    """
    time_candidates = ["TIME", "Time", "ThoiGian", "THOIGIAN", "Gio", "GIO"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    
    if not time_col:
        raise ValueError(f"Không tìm thấy cột thời gian trong: {df.columns.tolist()}")

    s_time = df[time_col].astype(str).str.strip()

    # Thử parse trực tiếp
    parsed = pd.to_datetime(s_time, errors="coerce")
    if parsed.notna().mean() >= 0.8:
        return parsed

    # Trường hợp ngày ở cột riêng
    date_candidates = ["DATE", "Date", "Ngay", "NGAY", "Ngày"]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if date_col:
        s_date = pd.to_datetime(df[date_col], errors="coerce")
        # Chuẩn hóa chuỗi thời gian nếu dạng mm:ss
        s_time_fixed = s_time.str.replace(r"\s+", "", regex=True)
        s_time_fixed = s_time_fixed.where(s_time_fixed.str.count(":") == 2, "00:" + s_time_fixed)
        s_delta = pd.to_timedelta(s_time_fixed, errors="coerce")
        return s_date + s_delta

    return parsed


# =============================================================================
# 3. ĐỌC VÀ LÀM SẠCH SƠ BỘ TỪNG FILE
# =============================================================================
def load_and_standardize_single_file(file_path: Path) -> pd.DataFrame:
    """
    Đọc một file CSV đơn lẻ và chuẩn hóa cơ bản:
    - Loại bỏ khoảng trắng tên cột
    - Chuẩn hóa TIME thành Datetime
    - Chuyển đổi dữ liệu số, dấu phẩy thành dấu chấm
    """
    # Thử đọc với encoding utf-8, nếu lỗi đổi sang cp1252 / latin1
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, low_memory=False, encoding="latin1")

    # Xóa khoảng trắng tên cột
    df.columns = [str(c).strip() for c in df.columns]

    # Chuẩn hóa TIME
    df["TIME"] = parse_time_robust(df)
    df = df.dropna(subset=["TIME"]).copy()

    # Chuẩn hóa cột số
    present_num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    for c in present_num_cols:
        s = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce")

    # Chuẩn hóa MaDiemDo và TenDiemDo
    if "MaDiemDo" in df.columns:
        df["MaDiemDo"] = pd.to_numeric(df["MaDiemDo"], errors="coerce")
        df = df.dropna(subset=["MaDiemDo"]).copy()
        df["MaDiemDo"] = df["MaDiemDo"].astype(int)

    if "TenDiemDo" in df.columns:
        df["TenDiemDo"] = df["TenDiemDo"].astype(str).str.strip()

    df["source_file"] = file_path.name
    return df


# =============================================================================
# 4. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY CHECK)
# =============================================================================
def audit_data_quality(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Kiểm tra toàn diện chất lượng dữ liệu và xuất báo cáo quality_report.csv
    """
    total_rows = len(df)
    unique_points = df["MaDiemDo"].nunique() if "MaDiemDo" in df.columns else 0
    t_min = df["TIME"].min()
    t_max = df["TIME"].max()

    # 1. Bản ghi trùng lặp
    dup_keys = ["TIME", "MaDiemDo"] if "MaDiemDo" in df.columns else ["TIME"]
    num_duplicates = int(df.duplicated(subset=dup_keys).sum())

    # 2. Tỷ lệ thiếu từng cột
    missing_stats = df.isna().sum()
    missing_pct = (missing_stats / total_rows * 100).round(2)

    # 3. Kiểm tra mốc lệch chu kỳ 30 phút
    # Mốc chuẩn 30 phút có minute % 30 == 0 và second == 0
    is_aligned_30m = (df["TIME"].dt.minute % 30 == 0) & (df["TIME"].dt.second == 0)
    misaligned_count = int((~is_aligned_30m).sum())

    # 4. Phát hiện bất thường vật lý
    outliers = {}
    if "P_kW" in df.columns:
        outliers["P_kW_negative"] = int((df["P_kW"] < 0).sum())
        outliers["P_kW_excessive"] = int((df["P_kW"] > 30000).sum())
    if "UA" in df.columns:
        outliers["UA_negative"] = int((df["UA"] < 0).sum())
        outliers["UA_excessive"] = int((df["UA"] > 35000).sum())

    # Tạo DataFrame báo cáo chất lượng
    report_data = []
    for col in df.columns:
        report_data.append({
            "column_name": col,
            "data_type": str(df[col].dtype),
            "missing_count": int(missing_stats[col]),
            "missing_pct": float(missing_pct[col]),
            "unique_values": int(df[col].nunique())
        })
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(output_dir / "quality_report.csv", index=False, encoding="utf-8-sig")

    quality_summary = {
        "total_rows": total_rows,
        "unique_points": unique_points,
        "time_min": str(t_min),
        "time_max": str(t_max),
        "duplicate_rows": num_duplicates,
        "misaligned_30m_rows": misaligned_count,
        "outliers": outliers
    }

    return quality_summary


# =============================================================================
# 5. LÀM TRÒN VÀ TẠO LƯỚI THỜI GIAN 30 PHÚT CHUẨN XÁC
# =============================================================================
def resample_and_impute_30m_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quy chuẩn dữ liệu về lưới 30 phút đều đặn cho từng MaDiemDo:
    - Làm tròn TIME về mốc 30 phút gần nhất.
    - Reindex đầy đủ từ t_min đến t_max theo chu kỳ 30 phút.
    - Chỉ nội suy tuyến tính các khoảng khuyết ngắn (tối đa 2 bước = 1 tiếng).
    - Giữ nguyên NaN cho các khoảng khuyết dài hơn.
    - Đánh dấu cột is_imputed = 1 cho các giá trị được nội suy.
    """
    # 1. Làm tròn mốc thời gian về mốc 30 phút gần nhất (00:00, 00:30, ...)
    df["TIME_30m"] = df["TIME"].dt.round("30min")

    # 2. Loại bỏ trùng lặp sau khi làm tròn (giữ bản ghi đầu tiên)
    df_sorted = df.sort_values(by=["MaDiemDo", "TIME_30m", "TIME"]).reset_index(drop=True)
    df_dedup = df_sorted.drop_duplicates(subset=["MaDiemDo", "TIME_30m"], keep="first").copy()

    # Nhóm theo từng điểm đo để xử lý độc lập (tránh Data Leakage)
    resampled_list = []

    for point_id, group in df_dedup.groupby("MaDiemDo"):
        group = group.set_index("TIME_30m").sort_index()

        # Tạo lưới thời gian 30 phút liên tục từ start đến end của điểm đo này
        t_start = group.index.min()
        t_end = group.index.max()
        full_timeline = pd.date_range(start=t_start, end=t_end, freq="30min", name="TIME")

        # Lưu lại bản gốc để kiểm tra ô nào bị thiếu trước khi nội suy
        group_reindexed = group.reindex(full_timeline)
        group_reindexed["MaDiemDo"] = point_id
        if "TenDiemDo" in group.columns:
            group_reindexed["TenDiemDo"] = group["TenDiemDo"].dropna().iloc[0] if len(group["TenDiemDo"].dropna()) > 0 else f"Point_{point_id}"
        if "MBA" in group.columns:
            group_reindexed["MBA"] = group["MBA"].dropna().iloc[0] if len(group["MBA"].dropna()) > 0 else "Unknown"

        # Đánh dấu dữ liệu thiếu trước khi nội suy
        was_missing = group_reindexed["P_kW"].isna()

        # Áp dụng nội suy tuyến tính có kiểm soát (giới hạn tối đa 2 bước = 1 giờ)
        present_nums = [c for c in NUMERIC_COLS if c in group_reindexed.columns]
        imputed_vals = group_reindexed[present_nums].interpolate(method="time", limit=2, limit_direction="forward")

        # Cột is_imputed: True nếu ban đầu bị thiếu và sau nội suy có giá trị
        is_imputed = was_missing & imputed_vals["P_kW"].notna()
        group_reindexed[present_nums] = imputed_vals
        group_reindexed["is_imputed"] = is_imputed.astype(int)

        group_reindexed = group_reindexed.reset_index().rename(columns={"index": "TIME"})
        resampled_list.append(group_reindexed)

    df_cleaned_long = pd.concat(resampled_list, ignore_index=True)
    return df_cleaned_long


# =============================================================================
# 6. TẠO ĐẶC TRƯNG VẬT LÝ & CHUỖI THỜI GIAN (FEATURE ENGINEERING)
# =============================================================================
def engineer_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Sinh các đặc trưng:
    - U_avg, I_avg, U_max, U_min, I_max, I_min
    - S_kVA, độ mất cân bằng pha
    - Đặc trưng lịch (hour, dayofweek, dayofmonth, month, is_weekend)
    - Lags & Rolling thống kê (không rò rỉ dữ liệu tương lai)
    """
    df = df_long.copy()
    eps = 1e-6

    # 1. Đặc trưng điện 3 pha
    u_cols = [c for c in ["UA", "UB", "UC"] if c in df.columns]
    if len(u_cols) == 3:
        df["U_avg"] = df[u_cols].mean(axis=1)
        df["U_max"] = df[u_cols].max(axis=1)
        df["U_min"] = df[u_cols].min(axis=1)
        df["U_unbalance_pct"] = (df["U_max"] - df["U_min"]) / (df["U_avg"].abs() + eps) * 100

    i_cols = [c for c in ["IA", "IB", "IC"] if c in df.columns]
    if len(i_cols) == 3:
        df["I_avg"] = df[i_cols].mean(axis=1)
        df["I_max"] = df[i_cols].max(axis=1)
        df["I_min"] = df[i_cols].min(axis=1)
        df["I_unbalance_pct"] = (df["I_max"] - df["I_min"]) / (df["I_avg"].abs() + eps) * 100

    # Công suất biểu kiến S (kVA) = sqrt(P^2 + Q^2)
    if "P_kW" in df.columns and "Q_kVAr" in df.columns:
        df["S_kVA"] = np.sqrt(df["P_kW"].fillna(0)**2 + df["Q_kVAr"].fillna(0)**2)

    # 2. Đặc trưng thời gian
    df["hour"] = df["TIME"].dt.hour + df["TIME"].dt.minute / 60.0
    df["dayofweek"] = df["TIME"].dt.dayofweek  # 0 = Thứ 2, 6 = Chủ Nhật
    df["dayofmonth"] = df["TIME"].dt.day
    df["month"] = df["TIME"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Mã hóa chu kỳ lượng giác (Cyclical encoding)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # 3. Đặc trưng trễ (Lags) và Thống kê trượt (Rolling) theo từng MaDiemDo
    featured_list = []
    for point_id, group in df.groupby("MaDiemDo"):
        group = group.sort_values("TIME").copy()

        # Lags: 1 (30p trước), 2 (1h trước), 3 (1.5h trước), 48 (24h trước), 336 (7 ngày trước)
        for lag in [1, 2, 3, 48, 336]:
            group[f"lag_{lag}"] = group["P_kW"].shift(lag)

        # Rolling statistics: Phải shift(1) trước khi rolling để đảm bảo KHÔNG rò rỉ dữ liệu tương lai
        p_shifted = group["P_kW"].shift(1)
        group["rolling_mean_3"] = p_shifted.rolling(window=3, min_periods=1).mean()
        group["rolling_mean_6"] = p_shifted.rolling(window=6, min_periods=1).mean()
        group["rolling_mean_48"] = p_shifted.rolling(window=48, min_periods=1).mean()

        featured_list.append(group)

    df_featured = pd.concat(featured_list, ignore_index=True)
    return df_featured


# =============================================================================
# 7. XÁC ĐỊNH TRẠM LỚN VÀ THỐNG KÊ (STATION SUMMARY)
# =============================================================================
def generate_station_summary(df_clean: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Tổng hợp danh sách các điểm đo: Tên, Số dòng, P_kW trung bình, cực đại, tỷ lệ thiếu.
    Xuất ra station_summary.csv giúp người dùng chọn đúng trạm lớn.
    """
    summary = []
    for point_id, group in df_clean.groupby("MaDiemDo"):
        name = group["TenDiemDo"].iloc[0] if "TenDiemDo" in group.columns else f"Point_{point_id}"
        count = len(group)
        p_mean = group["P_kW"].mean()
        p_max = group["P_kW"].max()
        missing_rate = group["P_kW"].isna().mean() * 100

        summary.append({
            "MaDiemDo": point_id,
            "TenDiemDo": name,
            "total_records": count,
            "P_mean_kW": round(float(p_mean), 3) if pd.notna(p_mean) else None,
            "P_max_kW": round(float(p_max), 3) if pd.notna(p_max) else None,
            "missing_rate_pct": round(float(missing_rate), 2)
        })

    df_summary = pd.DataFrame(summary).sort_values(by="P_mean_kW", ascending=False).reset_index(drop=True)
    df_summary.to_csv(output_dir / "station_summary.csv", index=False, encoding="utf-8-sig")
    return df_summary


# =============================================================================
# 8. THIẾT LẬP BÀI TOÁN HỌC CÓ GIÁM SÁT (SUPERVISED LEARNING)
# =============================================================================
def build_supervised_dataset(df_featured: pd.DataFrame, target_point: int = None, threshold_kw: float = 0.5) -> pd.DataFrame:
    """
    Tạo biến mục tiêu dự báo sau 30 phút:
    - target_P_kW = P_kW.shift(-1) (giá trị thực tế ở mốc thời gian tiếp theo)
    - target_delta_P = target_P_kW - P_kW (mức tăng/giảm định lượng)
    - target_direction: TĂNG / GIẢM / KHÔNG ĐỔI
    """
    df = df_featured.copy()

    # Nếu người dùng chỉ định điểm đo trạm lớn, lọc theo điểm đó
    if target_point is not None:
        df = df[df["MaDiemDo"] == target_point].copy()

    supervised_list = []
    for point_id, group in df.groupby("MaDiemDo"):
        group = group.sort_values("TIME").copy()

        # Dịch chuyển ngược 1 bước để lấy giá trị sau 30 phút
        group["target_P_kW"] = group["P_kW"].shift(-1)

        # Loại bỏ các dòng không có target (dòng cuối cùng hoặc điểm bị gián đoạn)
        group = group.dropna(subset=["target_P_kW"]).copy()

        # Tính độ lệch tăng/giảm định lượng
        group["target_delta_P"] = group["target_P_kW"] - group["P_kW"]

        # Phân loại xu hướng
        conditions = [
            group["target_delta_P"] > threshold_kw,
            group["target_delta_P"] < -threshold_kw
        ]
        choices = ["TĂNG", "GIẢM"]
        group["target_direction"] = np.select(conditions, choices, default="KHÔNG ĐỔI")

        supervised_list.append(group)

    if not supervised_list:
        return pd.DataFrame()

    df_supervised = pd.concat(supervised_list, ignore_index=True)
    return df_supervised


# =============================================================================
# 9. PHÂN CHIA TẬP DỮ LIỆU THEO THỜI GIAN (TIME-SERIES SPLIT)
# =============================================================================
def time_series_split(df_supervised: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """
    Phân chia tuần tự theo thời gian:
    - Train: 70%
    - Validation: 15%
    - Test: 15%
    """
    df_sorted = df_supervised.sort_values("TIME").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    split_info = {
        "train_rows": len(train_df),
        "train_start": str(train_df["TIME"].min()),
        "train_end": str(train_df["TIME"].max()),
        "val_rows": len(val_df),
        "val_start": str(val_df["TIME"].min()),
        "val_end": str(val_df["TIME"].max()),
        "test_rows": len(test_df),
        "test_start": str(test_df["TIME"].min()),
        "test_end": str(test_df["TIME"].max()),
    }

    return train_df, val_df, test_df, split_info


# =============================================================================
# 10. VẼ 6 BIỂU ĐỒ KIỂM TRA CHẤT LƯỢNG (VALIDATION CHARTS)
# =============================================================================
def generate_all_plots(df_raw: pd.DataFrame, df_clean: pd.DataFrame, output_dir: Path):
    """
    Vẽ đầy đủ 6 biểu đồ yêu cầu:
    1. Công suất theo thời gian
    2. Tỷ lệ thiếu dữ liệu
    3. Phân phối P_kW
    4. So sánh dữ liệu gốc và sau khi nội suy
    5. Phụ tải trung bình theo giờ trong ngày
    6. Phụ tải trung bình theo ngày trong tuần
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_point = df_clean["MaDiemDo"].iloc[0]

    # --- 1. Công suất theo thời gian ---
    fig, ax = plt.subplots(figsize=(15, 6))
    sub = df_clean[df_clean["MaDiemDo"] == sample_point].set_index("TIME")
    # Lấy mẫu 1 tháng đầu để nhìn rõ nét
    sample_sub = sub.iloc[:1440]  # 1440 mốc = 30 ngày
    ax.plot(sample_sub.index, sample_sub["P_kW"], color="tab:blue", lw=1.2, label=f"Điểm đo {sample_point}")
    ax.set_title("1. Chuỗi công suất P_kW theo thời gian (Mẫu 30 ngày)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Công suất P (kW)")
    ax.legend()
    fig.savefig(output_dir / "01_power_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 2. Tỷ lệ thiếu dữ liệu ---
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_pct = (df_raw[NUMERIC_COLS].isna().mean() * 100).sort_values(ascending=False)
    sns.barplot(x=missing_pct.values, y=missing_pct.index, ax=ax, palette="Blues_r")
    ax.set_title("2. Tỷ lệ thiếu dữ liệu của các biến số (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tỷ lệ thiếu (%)")
    for i, v in enumerate(missing_pct.values):
        ax.text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=10)
    fig.savefig(output_dir / "02_missing_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 3. Phân phối P_kW ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    p_clean = df_clean["P_kW"].dropna()
    sns.histplot(p_clean, bins=50, kde=True, ax=axes[0], color="tab:green")
    axes[0].set_title("Phân phối tần suất P_kW", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("P (kW)")

    sns.boxplot(y=p_clean, ax=axes[1], color="lightgreen")
    axes[1].set_title("Boxplot công suất P_kW", fontsize=12, fontweight="bold")
    fig.suptitle("3. Phân phối và khoảng biến thiên của công suất P_kW", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "03_power_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 4. So sánh dữ liệu gốc và sau khi nội suy ---
    fig, ax = plt.subplots(figsize=(15, 6))
    raw_sub = df_raw[df_raw["MaDiemDo"] == sample_point].set_index("TIME").sort_index()
    clean_sub = df_clean[df_clean["MaDiemDo"] == sample_point].set_index("TIME").sort_index()

    # Mẫu 1 tuần
    t_start = clean_sub.index.min()
    t_end = t_start + pd.Timedelta(days=7)

    r_slice = raw_sub.loc[t_start:t_end]
    c_slice = clean_sub.loc[t_start:t_end]

    ax.plot(r_slice.index, r_slice["P_kW"], "r.", markersize=7, alpha=0.6, label="Dữ liệu gốc (Thô / Khuyết)")
    ax.plot(c_slice.index, c_slice["P_kW"], "b-", lw=1.2, alpha=0.8, label="Dữ liệu sau khi làm sạch & nội suy")
    ax.set_title("4. Đối chiếu dữ liệu gốc và sau khi nội suy (Khung thời gian 7 ngày)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("P (kW)")
    ax.legend()
    fig.savefig(output_dir / "04_imputation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 5. Phụ tải trung bình theo giờ trong ngày ---
    fig, ax = plt.subplots(figsize=(12, 5))
    df_clean["hour_int"] = df_clean["TIME"].dt.hour
    avg_hour = df_clean.groupby("hour_int")["P_kW"].mean()
    ax.plot(avg_hour.index, avg_hour.values, marker="o", color="tab:orange", lw=2)
    ax.set_title("5. Đường cong phụ tải trung bình theo 24 giờ trong ngày", fontsize=14, fontweight="bold")
    ax.set_xlabel("Giờ trong ngày (0 - 23)")
    ax.set_ylabel("Công suất P_kW trung bình")
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.savefig(output_dir / "05_avg_load_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 6. Phụ tải trung bình theo ngày trong tuần ---
    fig, ax = plt.subplots(figsize=(10, 5))
    weekday_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
    df_clean["dow"] = df_clean["TIME"].dt.dayofweek
    avg_dow = df_clean.groupby("dow")["P_kW"].mean()
    sns.barplot(x=[weekday_names[i] for i in avg_dow.index], y=avg_dow.values, ax=ax, palette="Purples_d")
    ax.set_title("6. Phụ tải trung bình theo ngày trong tuần", fontsize=14, fontweight="bold")
    ax.set_xlabel("Ngày trong tuần")
    ax.set_ylabel("Công suất P_kW trung bình")
    fig.savefig(output_dir / "06_avg_load_by_weekday.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# HÀM THỰC THI PIPELINE CHÍNH (MAIN PIPELINE)
# =============================================================================
def run_pipeline(
    data_dir_str: str,
    output_dir_str: str = "output_preprocessing",
    target_point: int = None,
    threshold_kw: float = 0.5
):
    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ DỮ LIỆU DỰ BÁO PHỤ TẢI ĐIỆN 30 PHÚT")
    print(f"Thư mục dữ liệu: {data_dir.resolve()}")
    print(f"Thư mục xuất kết quả: {output_dir.resolve()}")
    print("=" * 70)

    # 1. Tìm các file CSV hợp lệ
    csv_files = find_csv_files(data_dir)
    print(f"\n[Bước 1/10] Đã tìm thấy {len(csv_files)} file CSV đo đếm hợp lệ.")
    if not csv_files:
        raise SystemExit("Không tìm thấy file CSV nào để xử lý!")

    # 2 & 3. Đọc và chuẩn hóa dữ liệu số
    print("\n[Bước 2 & 3/10] Đang đọc, hợp nhất và chuẩn hóa kiểu dữ liệu...")
    dfs = []
    for f in csv_files:
        try:
            df_item = load_and_standardize_single_file(f)
            dfs.append(df_item)
        except Exception as e:
            print(f"  - Cảnh báo: Lỗi đọc file {f.name}: {e}")

    df_raw_merged = pd.concat(dfs, ignore_index=True)
    print(f"  -> Tổng số dòng dữ liệu thô đọc được: {len(df_raw_merged):,}")

    # 4. Kiểm tra chất lượng dữ liệu
    print("\n[Bước 4/10] Kiểm tra chất lượng dữ liệu và xuất quality_report.csv...")
    audit_summary = audit_data_quality(df_raw_merged, output_dir)
    print(f"  -> Số điểm đo: {audit_summary['unique_points']}")
    print(f"  -> Khoảng thời gian: {audit_summary['time_min']} đến {audit_summary['time_max']}")
    print(f"  -> Số dòng trùng lặp: {audit_summary['duplicate_rows']}")

    # 5 & 6. Tạo lưới thời gian 30 phút và nội suy kiểm soát
    print("\n[Bước 5 & 6/10] Đưa về lưới 30 phút chuẩn và thực hiện nội suy có kiểm soát...")
    df_cleaned_long = resample_and_impute_30m_grid(df_raw_merged)
    df_cleaned_long.to_csv(output_dir / "cleaned_long.csv", index=False, encoding="utf-8-sig")
    print(f"  -> Đã lưu: cleaned_long.csv ({len(df_cleaned_long):,} dòng)")

    # Pivot sang Bảng ngang (cleaned_wide.csv)
    print("  -> Đang chuyển đổi sang định dạng bảng ngang (cleaned_wide.csv)...")
    df_wide = df_cleaned_long.pivot_table(
        index="TIME",
        columns="MaDiemDo",
        values=["P_kW", "Q_kVAr"],
        aggfunc="first"
    )
    df_wide.columns = [f"{var}_{pt}" for (var, pt) in df_wide.columns]
    df_wide = df_wide.reset_index()
    df_wide.to_csv(output_dir / "cleaned_wide.csv", index=False, encoding="utf-8-sig")
    print(f"  -> Đã lưu: cleaned_wide.csv")

    # 7. Tạo đặc trưng (Feature Engineering)
    print("\n[Bước 7/10] Trích xuất đặc trưng vật lý 3 pha, chu kỳ lịch, Lags và Rolling...")
    df_featured = engineer_features(df_cleaned_long)
    df_featured.to_csv(output_dir / "featured_data.csv", index=False, encoding="utf-8-sig")
    print(f"  -> Đã lưu: featured_data.csv")

    # 8. Xác định trạm lớn (Station Summary)
    print("\n[Bước 8/10] Tổng hợp danh sách trạm và xuất station_summary.csv...")
    df_station_summary = generate_station_summary(df_cleaned_long, output_dir)
    print("  -> Danh sách các điểm đo tiêu biểu:")
    print(df_station_summary.to_string(index=False))

    # Tự động chọn trạm lớn nếu chưa chỉ định (chọn điểm có P_mean lớn nhất)
    if target_point is None:
        target_point = int(df_station_summary.iloc[0]["MaDiemDo"])
        print(f"  -> Tự động chọn trạm mục tiêu lớn nhất: MaDiemDo = {target_point} ({df_station_summary.iloc[0]['TenDiemDo']})")

    # 9. Tạo tập học có giám sát (Supervised Data)
    print(f"\n[Bước 9/10] Tạo biến mục tiêu dự báo sau 30 phút cho trạm {target_point}...")
    df_supervised = build_supervised_dataset(df_featured, target_point=target_point, threshold_kw=threshold_kw)
    df_supervised.to_csv(output_dir / "supervised_data.csv", index=False, encoding="utf-8-sig")
    print(f"  -> Đã lưu: supervised_data.csv ({len(df_supervised):,} mẫu)")

    # 10. Chia dữ liệu theo thời gian (Train 70% / Val 15% / Test 15%)
    print("\n[Bước 10/10] Chia dữ liệu tuần tự theo thời gian...")
    train_df, val_df, test_df, split_info = time_series_split(df_supervised, train_ratio=0.70, val_ratio=0.15)
    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(output_dir / "validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")
    print(f"  -> Train set:      {split_info['train_rows']:,} dòng ({split_info['train_start']} -> {split_info['train_end']})")
    print(f"  -> Validation set: {split_info['val_rows']:,} dòng ({split_info['val_start']} -> {split_info['val_end']})")
    print(f"  -> Test set:       {split_info['test_rows']:,} dòng ({split_info['test_start']} -> {split_info['test_end']})")

    # Vẽ 6 biểu đồ kiểm tra
    print("\n[Đồ thị] Tự động sinh 6 biểu đồ kiểm tra chất lượng dữ liệu...")
    generate_all_plots(df_raw_merged, df_cleaned_long, output_dir)
    print(f"  -> Đã lưu toàn bộ biểu đồ vào thư mục: {output_dir.resolve()}")

    # Xuất file JSON tổng kết
    summary_final = {
        "timestamp": datetime.now().isoformat(),
        "total_files_read": len(csv_files),
        "raw_rows": len(df_raw_merged),
        "cleaned_rows": len(df_cleaned_long),
        "unique_points": int(df_cleaned_long["MaDiemDo"].nunique()),
        "selected_target_point": target_point,
        "split_info": split_info,
        "missing_rate_before_pct": round(float(df_raw_merged["P_kW"].isna().mean() * 100), 2),
        "missing_rate_after_pct": round(float(df_cleaned_long["P_kW"].isna().mean() * 100), 2),
    }

    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_final, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH PIPELINE TIỀN XỬ LÝ THÀNH CÔNG!")
    print("=" * 70)
    return summary_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline tiền xử lý dữ liệu phụ tải điện 30 phút")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn thư mục chứa dữ liệu")
    parser.add_argument("--output_dir", type=str, default="output_preprocessing", help="Thư mục xuất kết quả")
    parser.add_argument("--target_point", type=int, default=None, help="Mã điểm đo trạm lớn cần dự báo")
    parser.add_argument("--threshold_kw", type=float, default=0.5, help="Ngưỡng biến thiên kW để phân loại TĂNG/GIẢM")
    args = parser.parse_args()

    # Tự động nhận diện đường dẫn (Colab vs Local)
    if args.data_dir:
        input_path = args.data_dir
    elif Path("/content/drive/MyDrive/Tam_TLU/datasets").exists():
        input_path = "/content/drive/MyDrive/Tam_TLU/datasets"
    elif Path("./datasets").exists():
        input_path = "./datasets"
    else:
        input_path = "."

    run_pipeline(
        data_dir_str=input_path,
        output_dir_str=args.output_dir,
        target_point=args.target_point,
        threshold_kw=args.threshold_kw
    )
