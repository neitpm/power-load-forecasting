# -*- coding: utf-8 -*-
"""
=============================================================================
PIPELINE TIỀN XỬ LÝ DỮ LIỆU DỰ BÁO PHỤ TẢI ĐIỆN 30 PHÚT (BẢN CHUẨN XÁC THỜI GIAN)
=============================================================================
Khắc phục triệt để 3 vấn đề lớn:
1. Loại bỏ hoàn toàn mốc thời gian bất thường sau 30/08/2025 (khống chế chặt dải 2020 - 2025).
2. Tách bạch rõ 3 mức độ thiếu:
   - Thiếu trong dữ liệu thô (raw missing).
   - Thiếu do tạo lưới 30 phút (grid gaps).
   - Thiếu còn lại sau khi nội suy có kiểm soát (<= 2 bước).
3. Chia Train/Validation/Test TRƯỚC KHI NỘI SUY (Chỉ nội suy độc lập trong từng tập,
   triệt tiêu hoàn toàn nguy cơ rò rỉ dữ liệu qua ranh giới tập dữ liệu - Zero Leakage).
=============================================================================
"""

import os
import sys
import re
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình thẩm mỹ đồ thị
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11

NUMERIC_COLS = ["UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr", "P_MW", "Q_MVAr"]
MIN_VALID_DATE = pd.to_datetime("2020-01-01 00:00:00")
MAX_VALID_DATE = pd.to_datetime("2025-08-31 23:59:59")


# =============================================================================
# 1. TÌM KIẾM VÀ ĐỌC FILE CSV HỢP LỆ
# =============================================================================
def find_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {data_dir}")

    all_csvs = sorted(data_dir.rglob("*.csv"))
    exclude_keywords = ["report", "summary", "cleaned", "featured", "supervised", "train", "validation", "test", "merged_with_features"]
    valid_csvs = [p for p in all_csvs if not any(kw in p.name.lower() for kw in exclude_keywords)]
    return valid_csvs


# =============================================================================
# 2. CHUẨN HÓA THỜI GIAN VÀ DỮ LIỆU SỐ (CHẶT CHẼ, CHỐNG NHẦM NĂM)
# =============================================================================
def parse_time_robust(df: pd.DataFrame) -> pd.Series:
    """
    Parse thời gian chặt chẽ:
    - Bắt buộc chuỗi thời gian phải chứa năm 2020-2025.
    - Tuyệt đối không dùng pd.to_datetime tự động điền ngày hiện tại (2026) khi chuỗi chỉ có mm:ss.
    """
    time_candidates = ["TIME", "Time", "ThoiGian", "THOIGIAN", "Gio", "GIO"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    
    if not time_col:
        return pd.Series([pd.NaT] * len(df))

    s_time = df[time_col].astype(str).str.strip()

    # Kiểm tra xem chuỗi có chứa năm 4 chữ số (2020-2025) hay không
    has_year = s_time.str.contains(r"\b202[0-5]\b", regex=True)

    # Nếu chuỗi không có năm, kiểm tra xem có cột DATE/Ngay đi kèm không
    date_candidates = ["DATE", "Date", "Ngay", "NGAY", "Ngày"]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if date_col:
        s_date = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        s_time_fixed = s_time.str.replace(r"\s+", "", regex=True)
        s_time_fixed = s_time_fixed.where(s_time_fixed.str.count(":") == 2, "00:" + s_time_fixed)
        dt = s_date + pd.to_timedelta(s_time_fixed, errors="coerce")
        return dt

    # Nếu không có cột DATE và s_time cũng không chứa năm 202x -> coi như NaT
    # (Tránh trường hợp pandas tự điền năm 2026 hiện tại)
    parsed = pd.Series([pd.NaT] * len(df), index=df.index)
    if has_year.any():
        parsed.loc[has_year] = pd.to_datetime(s_time.loc[has_year], errors="coerce", format="mixed")

    return parsed


def clean_single_numeric_series(series: pd.Series) -> pd.Series:
    """
    Xử lý số thập phân tiếng Việt (dấu phẩy), loại bỏ dấu chấm phân tách hàng nghìn nếu có.
    """
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    # Nếu có nhiều hơn 1 dấu chấm (ví dụ: '79.760.600') -> định dạng ngăn cách nghìn
    s = s.apply(lambda x: ".".join(x.split(".")[:2]) if x.count(".") > 1 else x)
    return pd.to_numeric(s, errors="coerce")


def load_and_standardize_single_file(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, low_memory=False, encoding="latin1")

    df.columns = [str(c).strip() for c in df.columns]

    # Parse TIME chặt chẽ
    df["TIME"] = parse_time_robust(df)
    df = df.dropna(subset=["TIME"]).copy()

    # Khống chế thời gian thực tế: Chỉ nhận dữ liệu từ 2020-01-01 đến 2025-08-31
    df = df[(df["TIME"] >= MIN_VALID_DATE) & (df["TIME"] <= MAX_VALID_DATE)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    raw_num_cols = ["UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr"]
    for c in raw_num_cols:
        if c in df.columns:
            df[c] = clean_single_numeric_series(df[c])

    # CHUẨN HÓA ĐƠN VỊ VẬT LÝ CHÍNH XÁC:
    # Giá trị trong cột gốc 'P_kW' thực chất là Megawatt (MW) (ví dụ: 9.3408 MW ~ 9,340.8 kW).
    # Giá trị trong cột gốc 'Q_kVAr' thực chất là MVAr (ví dụ: 1.156 MVAr ~ 1,156.0 kVAr).
    # Chuẩn hóa tạo cả 2 biến để báo cáo và mô hình không bị nhầm lẫn bậc độ lớn:
    if "P_kW" in df.columns:
        df["P_MW"] = df["P_kW"].copy()
        df["P_kW"] = df["P_MW"] * 1000.0
    if "Q_kVAr" in df.columns:
        df["Q_MVAr"] = df["Q_kVAr"].copy()
        df["Q_kVAr"] = df["Q_MVAr"] * 1000.0

    if "MaDiemDo" in df.columns:
        df["MaDiemDo"] = pd.to_numeric(df["MaDiemDo"], errors="coerce")
        df = df.dropna(subset=["MaDiemDo"]).copy()
        df["MaDiemDo"] = df["MaDiemDo"].astype(int)

    if "TenDiemDo" in df.columns:
        df["TenDiemDo"] = df["TenDiemDo"].astype(str).str.strip()

    df["source_file"] = file_path.name
    return df


# =============================================================================
# 3. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU THÔ (RAW QUALITY REPORT)
# =============================================================================
def audit_data_quality(df: pd.DataFrame, reports_dir: Path) -> dict:
    total_rows = len(df)
    unique_points = df["MaDiemDo"].nunique() if "MaDiemDo" in df.columns else 0
    t_min = df["TIME"].min()
    t_max = df["TIME"].max()

    dup_count = int(df.duplicated(subset=["TIME", "MaDiemDo"]).sum())
    missing_stats = df.isna().sum()
    missing_pct = (missing_stats / total_rows * 100).round(2)

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
    df_report.to_csv(reports_dir / "quality_report.csv", index=False, encoding="utf-8-sig")

    return {
        "total_rows": total_rows,
        "unique_points": unique_points,
        "time_min": str(t_min),
        "time_max": str(t_max),
        "duplicate_rows": dup_count,
        "missing_pct_P_kW": float(missing_pct.get("P_kW", 0.0))
    }


# =============================================================================
# 4. TẠO LƯỚI THỜI GIAN 30 PHÚT CHUẨN (KHÔNG NỘI SUY MÙ)
# =============================================================================
def build_regular_30m_grid(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Quy chuẩn dữ liệu về lưới 30 phút đều đặn:
    - Làm tròn TIME về mốc 30 phút gần nhất (00:00, 00:30, ...)
    - Khử trùng lặp trên cặp (MaDiemDo, TIME_30m)
    - Reindex đầy đủ từ t_min đến t_max theo chu kỳ 30 phút (CHƯA NỘI SUY).
    """
    df = df_raw.copy()
    df["TIME_30m"] = df["TIME"].dt.round("30min")
    df_sorted = df.sort_values(by=["MaDiemDo", "TIME_30m", "TIME"]).reset_index(drop=True)
    df_dedup = df_sorted.drop_duplicates(subset=["MaDiemDo", "TIME_30m"], keep="first").copy()

    # Dải thời gian chung từ min đến max thực tế
    t_start = df_dedup["TIME_30m"].min()
    t_end = df_dedup["TIME_30m"].max()
    full_timeline = pd.date_range(start=t_start, end=t_end, freq="30min", name="TIME")

    grid_list = []
    for point_id, group in df_dedup.groupby("MaDiemDo"):
        if "TIME" in group.columns:
            group = group.drop(columns=["TIME"])
        group = group.set_index("TIME_30m").sort_index()

        group_reindexed = group.reindex(full_timeline)
        group_reindexed["MaDiemDo"] = point_id
        if "TenDiemDo" in group.columns:
            group_reindexed["TenDiemDo"] = group["TenDiemDo"].dropna().iloc[0] if len(group["TenDiemDo"].dropna()) > 0 else f"Point_{point_id}"
        if "MBA" in group.columns:
            group_reindexed["MBA"] = group["MBA"].dropna().iloc[0] if len(group["MBA"].dropna()) > 0 else "Unknown"

        # Đánh dấu dữ liệu thiếu trên lưới trước khi can thiệp
        group_reindexed["was_raw_missing"] = group_reindexed["P_kW"].isna().astype(int)
        grid_list.append(group_reindexed.reset_index())

    df_grid = pd.concat(grid_list, ignore_index=True)
    return df_grid


# =============================================================================
# 5. CHIA TẬP TRAIN / VAL / TEST TRƯỚC KHI NỘI SUY (CHỐNG RÒ RỈ DỮ LIỆU)
# =============================================================================
def split_timeline_into_subsets(df_grid: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """
    Cắt tập dữ liệu theo mốc thời gian tuyệt đối trước khi bất kỳ thao tác nội suy nào diễn ra.
    """
    timeline_unique = pd.Series(df_grid["TIME"].unique()).sort_values().reset_index(drop=True)
    total_slots = len(timeline_unique)

    train_end_idx = int(total_slots * train_ratio)
    val_end_idx = int(total_slots * (train_ratio + val_ratio))

    t_train_cutoff = timeline_unique.iloc[train_end_idx - 1]
    t_val_start = timeline_unique.iloc[train_end_idx]
    t_val_cutoff = timeline_unique.iloc[val_end_idx - 1]
    t_test_start = timeline_unique.iloc[val_end_idx]

    train_mask = df_grid["TIME"] <= t_train_cutoff
    val_mask = (df_grid["TIME"] >= t_val_start) & (df_grid["TIME"] <= t_val_cutoff)
    test_mask = df_grid["TIME"] >= t_test_start

    train_subset = df_grid[train_mask].copy()
    val_subset = df_grid[val_mask].copy()
    test_subset = df_grid[test_mask].copy()

    split_meta = {
        "train_start": str(train_subset["TIME"].min()),
        "train_end": str(train_subset["TIME"].max()),
        "val_start": str(val_subset["TIME"].min()),
        "val_end": str(val_subset["TIME"].max()),
        "test_start": str(test_subset["TIME"].min()),
        "test_end": str(test_subset["TIME"].max()),
    }

    return train_subset, val_subset, test_subset, split_meta


# =============================================================================
# 6. NỘI SUY CÓ KIỂM SOÁT ĐỘC LẬP TRONG TỪNG TẬP (ISOLATED IMPUTATION)
# =============================================================================
def impute_isolated_subset(df_subset: pd.DataFrame, limit_steps: int = 2) -> pd.DataFrame:
    """
    Nội suy tuyến tính CHỈ trong phạm vi nội bộ của tập dữ liệu đó:
    - Không nhìn sang tương lai của tập kế tiếp.
    - Giới hạn tối đa limit_steps (2 bước = 1 tiếng).
    - Các khoảng trống dài hơn giữ nguyên NaN.
    """
    imputed_groups = []
    for point_id, group in df_subset.groupby("MaDiemDo"):
        group = group.set_index("TIME").sort_index().copy()
        present_nums = [c for c in NUMERIC_COLS if c in group.columns]

        was_missing = group["P_kW"].isna()
        # Nội suy có kiểm soát: method='time', limit=limit_steps, limit_direction='forward'
        imputed_vals = group[present_nums].interpolate(method="time", limit=limit_steps, limit_direction="forward")
        is_imputed = was_missing & imputed_vals["P_kW"].notna()

        group[present_nums] = imputed_vals
        group["is_imputed"] = is_imputed.astype(int)
        imputed_groups.append(group.reset_index())

    return pd.concat(imputed_groups, ignore_index=True)


# =============================================================================
# 7. TRÍCH XUẤT ĐẶC TRƯNG CHUỖI THỜI GIAN (FEATURE ENGINEERING)
# =============================================================================
def engineer_features(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()
    eps = 1e-6

    # Đặc trưng 3 pha
    if set(["UA", "UB", "UC"]).issubset(df.columns):
        df["U_avg"] = df[["UA", "UB", "UC"]].mean(axis=1)
        df["U_max"] = df[["UA", "UB", "UC"]].max(axis=1)
        df["U_min"] = df[["UA", "UB", "UC"]].min(axis=1)
        df["U_unbalance_pct"] = (df["U_max"] - df["U_min"]) / (df["U_avg"].abs() + eps) * 100

    if set(["IA", "IB", "IC"]).issubset(df.columns):
        df["I_avg"] = df[["IA", "IB", "IC"]].mean(axis=1)
        df["I_max"] = df[["IA", "IB", "IC"]].max(axis=1)
        df["I_min"] = df[["IA", "IB", "IC"]].min(axis=1)
        df["I_unbalance_pct"] = (df["I_max"] - df["I_min"]) / (df["I_avg"].abs() + eps) * 100

    if "P_kW" in df.columns and "Q_kVAr" in df.columns:
        df["S_kVA"] = np.sqrt(df["P_kW"].fillna(0)**2 + df["Q_kVAr"].fillna(0)**2)
        df["S_MVA"] = df["S_kVA"] / 1000.0

    df["hour"] = df["TIME"].dt.hour + df["TIME"].dt.minute / 60.0
    df["dayofweek"] = df["TIME"].dt.dayofweek
    df["dayofmonth"] = df["TIME"].dt.day
    df["month"] = df["TIME"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Lags & Rolling không rò rỉ dữ liệu (shift 1)
    feat_groups = []
    for point_id, grp in df.groupby("MaDiemDo"):
        grp = grp.sort_values("TIME").copy()
        for lag in [1, 2, 3, 48, 336]:
            grp[f"lag_{lag}"] = grp["P_kW"].shift(lag)
            if "P_MW" in grp.columns:
                grp[f"lag_mw_{lag}"] = grp["P_MW"].shift(lag)
        p_prev = grp["P_kW"].shift(1)
        grp["rolling_mean_3"] = p_prev.rolling(3, min_periods=1).mean()
        grp["rolling_mean_6"] = p_prev.rolling(6, min_periods=1).mean()
        grp["rolling_mean_48"] = p_prev.rolling(48, min_periods=1).mean()
        if "P_MW" in grp.columns:
            p_mw_prev = grp["P_MW"].shift(1)
            grp["rolling_mean_mw_3"] = p_mw_prev.rolling(3, min_periods=1).mean()
            grp["rolling_mean_mw_6"] = p_mw_prev.rolling(6, min_periods=1).mean()
            grp["rolling_mean_mw_48"] = p_mw_prev.rolling(48, min_periods=1).mean()
        feat_groups.append(grp)

    return pd.concat(feat_groups, ignore_index=True)


# =============================================================================
# 8. BẢNG TỔNG HỢP TRẠM VÀ PHÂN TÍCH RÕ NGUYÊN NHÂN THIẾU
# =============================================================================
def generate_station_summary(df_raw: pd.DataFrame, df_grid: pd.DataFrame, df_clean: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    """
    Phân tách rạch ròi 3 mức độ thiếu:
    1. Thiếu trong file gốc (Raw Missing P_kW).
    2. Thiếu do tạo lưới 30m liên tục (Grid Gap Missing).
    3. Thiếu còn lại sau nội suy <= 2 bước (Remaining Missing).
    Đồng thời hiển thị công suất theo cả 2 đơn vị chuẩn: kW và MW.
    """
    summary = []
    total_slots = df_grid["TIME"].nunique()

    for pid in sorted(df_grid["MaDiemDo"].unique()):
        raw_sub = df_raw[df_raw["MaDiemDo"] == pid]
        grid_sub = df_grid[df_grid["MaDiemDo"] == pid]
        clean_sub = df_clean[df_clean["MaDiemDo"] == pid]

        name = grid_sub["TenDiemDo"].iloc[0] if "TenDiemDo" in grid_sub.columns else f"Point_{pid}"
        raw_rows = len(raw_sub)
        raw_nulls = raw_sub["P_kW"].isna().sum() if "P_kW" in raw_sub.columns else 0
        raw_null_pct = (raw_nulls / raw_rows * 100) if raw_rows > 0 else 0

        grid_missing = grid_sub["P_kW"].isna().sum()
        grid_missing_pct = (grid_missing / total_slots) * 100

        clean_missing = clean_sub["P_kW"].isna().sum()
        clean_missing_pct = (clean_missing / total_slots) * 100

        imputed_count = int(clean_sub["is_imputed"].sum())
        p_mean_kw = clean_sub["P_kW"].mean()
        p_max_kw = clean_sub["P_kW"].max()
        p_mean_mw = clean_sub["P_MW"].mean() if "P_MW" in clean_sub.columns else (p_mean_kw / 1000.0 if pd.notna(p_mean_kw) else None)
        p_max_mw = clean_sub["P_MW"].max() if "P_MW" in clean_sub.columns else (p_max_kw / 1000.0 if pd.notna(p_max_kw) else None)

        summary.append({
            "MaDiemDo": pid,
            "TenDiemDo": name,
            "raw_recorded_rows": raw_rows,
            "raw_null_pct": round(float(raw_null_pct), 2),
            "total_30m_grid_slots": total_slots,
            "grid_missing_pct": round(float(grid_missing_pct), 2),
            "imputed_steps_count": imputed_count,
            "remaining_missing_pct": round(float(clean_missing_pct), 2),
            "P_mean_kW": round(float(p_mean_kw), 1) if pd.notna(p_mean_kw) else None,
            "P_max_kW": round(float(p_max_kw), 1) if pd.notna(p_max_kw) else None,
            "P_mean_MW": round(float(p_mean_mw), 3) if pd.notna(p_mean_mw) else None,
            "P_max_MW": round(float(p_max_mw), 3) if pd.notna(p_max_mw) else None
        })

    df_summary = pd.DataFrame(summary).sort_values(by="P_mean_kW", ascending=False).reset_index(drop=True)
    df_summary.to_csv(reports_dir / "station_summary.csv", index=False, encoding="utf-8-sig")
    return df_summary


# =============================================================================
# 9. TẠO BIẾN MỤC TIÊU CHO BÀI TOÁN HỌC CÓ GIÁM SÁT
# =============================================================================
def build_supervised_subset(df_featured_subset: pd.DataFrame, target_point: int, threshold_kw: float = 50.0):
    """
    Tạo biến mục tiêu:
    - target_P_kW: Công suất tại thời điểm t + 30 phút (kW)
    - target_P_MW: Công suất tại thời điểm t + 30 phút (MW)
    - target_delta_P_kW: Mức biến thiên Delta P = P[t+1] - P[t] (kW)
    - target_direction: Phân loại TĂNG / GIẢM / KHÔNG ĐỔI dựa trên ngưỡng biến thiên threshold_kw
      (Mặc định 50 kW ~ 0.05 MW, tương đương ~0.6% công suất trung bình trạm 8,000 kW)
    """
    sub = df_featured_subset[df_featured_subset["MaDiemDo"] == target_point].sort_values("TIME").copy()

    # Nhãn mục tiêu P[t+1] theo cả kW và MW
    sub["target_P_kW"] = sub["P_kW"].shift(-1)
    if "P_MW" in sub.columns:
        sub["target_P_MW"] = sub["P_MW"].shift(-1)
    else:
        sub["target_P_MW"] = sub["target_P_kW"] / 1000.0

    # Loại bỏ dòng không có target (dòng cuối cùng của tập)
    sub = sub.dropna(subset=["target_P_kW"]).copy()

    # Mức biến thiên và phân loại xu hướng
    sub["target_delta_P_kW"] = sub["target_P_kW"] - sub["P_kW"]
    sub["target_delta_P_MW"] = sub["target_P_MW"] - sub["P_MW"]
    sub["target_delta_P"] = sub["target_delta_P_kW"]  # Biến delta mặc định theo kW

    conditions = [
        sub["target_delta_P_kW"] > threshold_kw,
        sub["target_delta_P_kW"] < -threshold_kw
    ]
    sub["target_direction"] = np.select(conditions, ["TĂNG", "GIẢM"], default="KHÔNG ĐỔI")
    return sub


# =============================================================================
# 10. BỘ 8 BIỂU ĐỒ ĐÁNH GIÁ CHẤT LƯỢNG CAO (200 DPI - CHUẨN KHOA HỌC)
# =============================================================================
def generate_evaluation_figures(df_raw: pd.DataFrame, df_clean: pd.DataFrame, df_supervised: pd.DataFrame, target_point: int, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    clean_target = df_clean[df_clean["MaDiemDo"] == target_point].set_index("TIME").sort_index()
    raw_target = df_raw[df_raw["MaDiemDo"] == target_point].set_index("TIME").sort_index()

    # --- 1. Toàn cảnh chuỗi phụ tải & Đối chiếu phục hồi nội suy ---
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(16, 9))
    sample_month = clean_target.iloc[:1440]  # 30 ngày mẫu
    ax1a.plot(sample_month.index, sample_month["P_kW"], color="#1f77b4", lw=1.2, label=f"Điểm đo {target_point} (30 ngày)")
    ax1a.set_title(f"1A. Toàn cảnh chuỗi phụ tải P (kW) sau làm sạch (Khung 30 ngày - Điểm đo {target_point})", fontsize=13, fontweight="bold")
    ax1a.set_ylabel("Công suất P (kW)")
    ax1a.legend(loc="upper right")

    t0 = clean_target.index.min()
    t1 = t0 + pd.Timedelta(days=7)
    r_slice = raw_target.loc[t0:t1]
    c_slice = clean_target.loc[t0:t1]
    ax1b.plot(r_slice.index, r_slice["P_kW"], "r.", markersize=8, alpha=0.6, label="Dữ liệu gốc (Bị thiếu / Lệch giây)")
    ax1b.plot(c_slice.index, c_slice["P_kW"], color="#2ca02c", lw=1.5, alpha=0.9, label="Đã quy chuẩn lưới 30m & Nội suy ngắn (<=2 bước)")
    ax1b.set_title("1B. Kiểm chứng phục hồi dữ liệu: So sánh Gốc vs Sau xử lý (Zoom 7 ngày)", fontsize=13, fontweight="bold")
    ax1b.set_ylabel("Công suất P (kW)")
    ax1b.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(figures_dir / "01_load_timeline_and_imputation.png", dpi=200)
    plt.close(fig1)

    # --- 2. Tỷ lệ khuyết dữ liệu gốc (%) ---
    fig2, ax2 = plt.subplots(figsize=(11, 4.5))
    missing_cols = [c for c in ["UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr"] if c in df_raw.columns]
    missing_p = (df_raw[missing_cols].isna().mean() * 100).sort_values(ascending=True)
    bars2 = ax2.barh(missing_p.index, missing_p.values, color=sns.color_palette("Blues_r", len(missing_p)))
    ax2.set_title("2. Đánh giá tỷ lệ khuyết dữ liệu gốc theo từng biến số (%)", fontsize=13, fontweight="bold")
    for b in bars2:
        w = b.get_width()
        ax2.text(w + 0.05, b.get_y() + b.get_height()/2, f"{w:.2f}%", va="center", fontsize=10, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(figures_dir / "02_data_completeness_missing_matrix.png", dpi=200)
    plt.close(fig2)

    # --- 3. Phân phối xác suất & Boxplot ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 5.5))
    p_data = clean_target["P_kW"].dropna()
    p_mean_val = p_data.mean()
    p_med_val = p_data.median()
    sns.histplot(p_data, bins=60, kde=True, ax=axes3[0], color="#1f77b4", edgecolor="black", alpha=0.6)
    axes3[0].axvline(p_mean_val, color="red", linestyle="--", lw=2, label=f"Mean = {p_mean_val:,.1f} kW ({p_mean_val/1000:.2f} MW)")
    axes3[0].axvline(p_med_val, color="green", linestyle="-.", lw=2, label=f"Median = {p_med_val:,.1f} kW ({p_med_val/1000:.2f} MW)")
    axes3[0].set_title("Phân phối xác suất (Histogram & KDE) của P (kW)", fontsize=12, fontweight="bold")
    axes3[0].set_xlabel("Công suất P (kW)")
    axes3[0].legend()

    sns.boxplot(y=p_data, ax=axes3[1], color="#98df8a")
    axes3[1].set_title("Biểu đồ Boxplot: Khoảng tứ phân vị & Ngoại lai của P (kW)", fontsize=12, fontweight="bold")
    axes3[1].set_ylabel("Công suất P (kW)")
    fig3.suptitle("3. Kiểm định phân phối và biên độ công suất thực tế P (kW)", fontsize=15, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(figures_dir / "03_power_distribution_and_anomaly.png", dpi=200)
    plt.close(fig3)

    # --- 4. Ma trận tương quan đặc trưng ---
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    corr_cols = [c for c in ["P_kW", "target_P_kW", "lag_1", "lag_2", "lag_48", "rolling_mean_3", "U_avg", "I_avg", "S_kVA", "CosF"] if c in df_supervised.columns]
    sns.heatmap(df_supervised[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax4)
    ax4.set_title("4. Ma trận tương quan đặc trưng với biến mục tiêu (target_P_kW)", fontsize=13, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig(figures_dir / "04_correlation_heatmap_features.png", dpi=200)
    plt.close(fig4)

    # --- 5. Đánh giá biến mục tiêu (Delta P & Nhãn xu hướng) ---
    fig5, axes5 = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.histplot(df_supervised["target_delta_P_kW"].dropna(), bins=70, kde=True, ax=axes5[0], color="#9467bd")
    axes5[0].axvline(0, color="red", linestyle="--", lw=1.5)
    axes5[0].set_title("Phân phối Delta P = P[t+1] - P[t] (kW)", fontsize=12, fontweight="bold")
    axes5[0].set_xlabel("Delta P (kW)")

    d_counts = df_supervised["target_direction"].value_counts()
    bars5 = axes5[1].bar(d_counts.index, d_counts.values, color=["#2ca02c", "#d62728", "#7f7f7f"][:len(d_counts)], edgecolor="black")
    axes5[1].set_title("Tỷ lệ nhãn xu hướng (Ngưỡng biến thiên |Delta P| >= 50 kW)", fontsize=12, fontweight="bold")
    for b in bars5:
        h = b.get_height()
        axes5[1].text(b.get_x() + b.get_width()/2, h + 10, f"{h:,}\n({h/len(df_supervised)*100:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    fig5.suptitle("5. Kiểm định biến mục tiêu dự báo (Target Verification)", fontsize=15, fontweight="bold")
    fig5.tight_layout()
    fig5.savefig(figures_dir / "05_target_evaluation_and_direction.png", dpi=200)
    plt.close(fig5)

    # --- 6. Scatter P[t] vs P[t+1] ---
    fig6, ax6 = plt.subplots(figsize=(7, 7))
    samp = df_supervised.sample(n=min(3000, len(df_supervised)), random_state=42)
    ax6.scatter(samp["P_kW"], samp["target_P_kW"], alpha=0.3, color="#1f77b4", s=25)
    m_val = min(samp["P_kW"].min(), samp["target_P_kW"].min())
    mx_val = max(samp["P_kW"].max(), samp["target_P_kW"].max())
    ax6.plot([m_val, mx_val], [m_val, mx_val], "r--", lw=2, label="Đường y = x")
    r = df_supervised[["P_kW", "target_P_kW"]].corr().iloc[0, 1]
    ax6.set_title(f"6. Tương quan tuyến tính P[t] vs P[t+1] (r = {r:.4f})", fontsize=13, fontweight="bold")
    ax6.set_xlabel("Công suất hiện tại P[t] (kW)")
    ax6.set_ylabel("Công suất sau 30 phút P[t+1] (kW)")
    ax6.legend(loc="upper left")
    fig6.tight_layout()
    fig6.savefig(figures_dir / "06_scatter_lag_vs_target.png", dpi=200)
    plt.close(fig6)

    # --- 7. Tính mùa vụ Giờ & Thứ ---
    fig7, axes7 = plt.subplots(1, 2, figsize=(16, 5.5))
    h_m = clean_target.groupby(clean_target.index.hour)["P_kW"].mean()
    h_s = clean_target.groupby(clean_target.index.hour)["P_kW"].std()
    axes7[0].plot(h_m.index, h_m.values, marker="o", color="#ff7f0e", lw=2.5)
    axes7[0].fill_between(h_m.index, h_m.values - h_s.values, h_m.values + h_s.values, color="#ff7f0e", alpha=0.2)
    axes7[0].set_title("Phụ tải trung bình theo 24 giờ trong ngày (kW)", fontsize=12, fontweight="bold")
    axes7[0].set_xlabel("Giờ trong ngày (0 - 23)")
    axes7[0].set_ylabel("Công suất P trung bình (kW)")
    axes7[0].set_xticks(range(0, 24))

    dow_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
    d_m = clean_target.groupby(clean_target.index.dayofweek)["P_kW"].mean()
    axes7[1].bar(dow_names[:len(d_m)], d_m.values, color=sns.color_palette("Set2", len(d_m)), edgecolor="black")
    axes7[1].set_title("Phụ tải trung bình theo ngày trong tuần (kW)", fontsize=12, fontweight="bold")
    axes7[1].set_ylabel("Công suất P trung bình (kW)")
    fig7.suptitle("7. Đánh giá tính mùa vụ phụ tải điện (Seasonality Profile)", fontsize=15, fontweight="bold")
    fig7.tight_layout()
    fig7.savefig(figures_dir / "07_daily_and_weekly_seasonality.png", dpi=200)
    plt.close(fig7)

    # --- 8. Timeline chia tập Train - Val - Test (Thực tế đến tháng 8/2025) ---
    fig8, ax8 = plt.subplots(figsize=(15, 4))
    df_sup_sorted = df_supervised.sort_values("TIME").reset_index(drop=True)
    n = len(df_sup_sorted)
    t_idx = int(n * 0.70)
    v_idx = int(n * 0.85)

    ax8.plot(df_sup_sorted.iloc[:t_idx]["TIME"], df_sup_sorted.iloc[:t_idx]["P_kW"], color="#1f77b4", label="Train (70%)", lw=1)
    ax8.plot(df_sup_sorted.iloc[t_idx:v_idx]["TIME"], df_sup_sorted.iloc[t_idx:v_idx]["P_kW"], color="#ff7f0e", label="Validation (15%)", lw=1)
    ax8.plot(df_sup_sorted.iloc[v_idx:]["TIME"], df_sup_sorted.iloc[v_idx:]["P_kW"], color="#2ca02c", label="Test (15%)", lw=1)
    ax8.set_title("8. Phân chia tập dữ liệu tuần tự theo thời gian (Thực tế: 03/2020 -> 08/2025)", fontsize=13, fontweight="bold")
    ax8.set_xlabel("Thời gian")
    ax8.set_ylabel("Công suất P (kW)")
    ax8.legend(loc="upper right")
    fig8.tight_layout()
    fig8.savefig(figures_dir / "08_data_split_timeline.png", dpi=200)
    plt.close(fig8)


# =============================================================================
# HÀM THỰC THI CHÍNH TOÀN BỘ PIPELINE
# =============================================================================
def run_pipeline(
    data_dir_str: str,
    output_dir_str: str = "output_preprocessing",
    target_point: int = None,
    threshold_kw: float = 50.0
):
    base_out = Path(output_dir_str)
    data_out = base_out / "data"
    reports_out = base_out / "reports"
    figures_out = base_out / "figures"

    for d in [data_out, reports_out, figures_out]:
        d.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir_str)

    print("=" * 80)
    print("PIPELINE TIỀN XỬ LÝ PHỤ TẢI ĐIỆN 30 PHÚT (BẢN SỬA LỖI THỜI GIAN & CHIA TẬP)")
    print(f"📁 Dữ liệu đầu vào:      {data_dir.resolve()}")
    print(f"💾 Thư mục dữ liệu sạch: {data_out.resolve()}")
    print(f"📑 Thư mục báo cáo CSV:  {reports_out.resolve()}")
    print(f"📊 Thư mục biểu đồ (DPI cao): {figures_out.resolve()}")
    print("=" * 80)

    # 1. Quét CSV & đọc chuẩn hóa
    csv_files = find_csv_files(data_dir)
    print(f"\n[1/7] Tìm thấy {len(csv_files)} file CSV hợp lệ. Đang chuẩn hóa...")
    dfs = []
    for f in csv_files:
        df_item = load_and_standardize_single_file(f)
        if len(df_item) > 0:
            dfs.append(df_item)

    df_raw = pd.concat(dfs, ignore_index=True)
    t_min_real = df_raw["TIME"].min()
    t_max_real = df_raw["TIME"].max()
    print(f"      -> Tổng số dòng dữ liệu thô hợp lệ: {len(df_raw):,}")
    print(f"      -> Dải thời gian thực tế: {t_min_real} đến {t_max_real}")

    # 2. Báo cáo chất lượng thô
    print("\n[2/7] Xuất báo cáo chất lượng: reports/quality_report.csv...")
    audit_summary = audit_data_quality(df_raw, reports_out)

    # 3. Tạo lưới 30 phút CHƯA NỘI SUY
    print("\n[3/7] Quy chuẩn lưới thời gian 30 phút đồng nhất cho từng điểm đo (CHƯA nội suy)...")
    df_grid = build_regular_30m_grid(df_raw)
    total_slots = df_grid["TIME"].nunique()
    print(f"      -> Tổng số mốc 30 phút chuẩn: {total_slots:,} mốc")

    # 4. CHIA TẬP TRAIN / VAL / TEST TRƯỚC KHI NỘI SUY (CHỐNG RÒ RỈ DỮ LIỆU)
    print("\n[4/7] Chia Train (70%) - Validation (15%) - Test (15%) THEO THỜI GIAN trước khi nội suy...")
    train_raw_grid, val_raw_grid, test_raw_grid, split_meta = split_timeline_into_subsets(df_grid, 0.70, 0.15)
    print(f"      -> Train subset: {len(train_raw_grid):,} dòng ({split_meta['train_start']} -> {split_meta['train_end']})")
    print(f"      -> Val subset:   {len(val_raw_grid):,} dòng ({split_meta['val_start']} -> {split_meta['val_end']})")
    print(f"      -> Test subset:  {len(test_raw_grid):,} dòng ({split_meta['test_start']} -> {split_meta['test_end']})")

    # 5. NỘI SUY ĐỘC LẬP TRONG TỪNG TẬP (ISOLATED IMPUTATION)
    print("\n[5/7] Nội suy kiểm soát (<= 2 bước) độc lập trong từng tập (Zero Data Leakage)...")
    train_clean = impute_isolated_subset(train_raw_grid, limit_steps=2)
    val_clean = impute_isolated_subset(val_raw_grid, limit_steps=2)
    test_clean = impute_isolated_subset(test_raw_grid, limit_steps=2)

    train_clean["split"] = "train"
    val_clean["split"] = "val"
    test_clean["split"] = "test"

    df_clean_long = pd.concat([train_clean, val_clean, test_clean], ignore_index=True)
    df_clean_long.to_csv(data_out / "cleaned_long.csv", index=False, encoding="utf-8-sig")

    df_wide = df_clean_long.pivot_table(index="TIME", columns="MaDiemDo", values=["P_kW", "P_MW", "Q_kVAr"], aggfunc="first")
    df_wide.columns = [f"{var}_{pt}" for (var, pt) in df_wide.columns]
    df_wide.reset_index().to_csv(data_out / "cleaned_wide.csv", index=False, encoding="utf-8-sig")

    # 6. Trích xuất đặc trưng & Báo cáo trạm
    print("\n[6/7] Trích xuất đặc trưng vật lý & chuỗi thời gian, xuất station_summary.csv...")
    df_featured = engineer_features(df_clean_long)
    df_featured.to_csv(data_out / "featured_data.csv", index=False, encoding="utf-8-sig")

    df_station_summary = generate_station_summary(df_raw, df_grid, df_clean_long, reports_out)
    if target_point is None:
        target_point = int(df_station_summary.iloc[0]["MaDiemDo"])
    print(f"      -> Trạm mục tiêu lớn nhất được chọn: MaDiemDo = {target_point} ({df_station_summary.iloc[0]['TenDiemDo']})")

    # Tạo tập học có giám sát và chia train / val / test
    feat_train = df_featured[df_featured["split"] == "train"]
    feat_val = df_featured[df_featured["split"] == "val"]
    feat_test = df_featured[df_featured["split"] == "test"]

    sup_train = build_supervised_subset(feat_train, target_point, threshold_kw)
    sup_val = build_supervised_subset(feat_val, target_point, threshold_kw)
    sup_test = build_supervised_subset(feat_test, target_point, threshold_kw)

    df_supervised = pd.concat([sup_train, sup_val, sup_test], ignore_index=True)
    df_supervised.to_csv(data_out / "supervised_data.csv", index=False, encoding="utf-8-sig")
    sup_train.to_csv(data_out / "train.csv", index=False, encoding="utf-8-sig")
    sup_val.to_csv(data_out / "validation.csv", index=False, encoding="utf-8-sig")
    sup_test.to_csv(data_out / "test.csv", index=False, encoding="utf-8-sig")

    print(f"      -> Supervised Data: Train {len(sup_train):,} | Val {len(sup_val):,} | Test {len(sup_test):,}")

    # 7. Sinh 8 biểu đồ đánh giá chuẩn khoa học vào figures/
    print("\n[7/7] Sinh 8 biểu đồ đánh giá chất lượng cao (DPI 200) vào figures/...")
    generate_evaluation_figures(df_raw, df_clean_long, df_supervised, target_point, figures_out)

    # Lưu metadata tổng kết
    summary_final = {
        "timestamp": datetime.now().isoformat(),
        "total_files_read": len(csv_files),
        "raw_rows": len(df_raw),
        "clean_rows": len(df_clean_long),
        "selected_target_point": target_point,
        "time_range": {
            "start": str(t_min_real),
            "end": str(t_max_real)
        },
        "split_info": {
            "train_samples": len(sup_train),
            "train_range": [split_meta["train_start"], split_meta["train_end"]],
            "val_samples": len(sup_val),
            "val_range": [split_meta["val_start"], split_meta["val_end"]],
            "test_samples": len(sup_test),
            "test_range": [split_meta["test_start"], split_meta["test_end"]],
        },
        "paths": {
            "clean_data_dir": str(data_out.resolve()),
            "reports_dir": str(reports_out.resolve()),
            "figures_dir": str(figures_out.resolve())
        }
    }
    with open(reports_out / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_final, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✅ ĐÃ HOÀN TẤT VÀ KHẮC PHỤC TRIỆT ĐỂ LỖI THỜI GIAN & CHIA TẬP KHÔNG RÒ RỈ!")
    print("=" * 80)
    return summary_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline tiền xử lý phụ tải điện 30 phút")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output_preprocessing")
    parser.add_argument("--target_point", type=int, default=None)
    parser.add_argument("--threshold_kw", type=float, default=50.0)
    args = parser.parse_args()

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
