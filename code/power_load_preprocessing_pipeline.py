# -*- coding: utf-8 -*-
"""
=============================================================================
PIPELINE TIỀN XỬ LÝ DỮ LIỆU DỰ BÁO PHỤ TẢI ĐIỆN 30 PHÚT (BẢN CHUẨN ĐÁNH GIÁ)
=============================================================================
Cấu trúc lưu trữ đầu ra tách bạch riêng biệt:
- output_preprocessing/data/    : Chứa dữ liệu sạch (cleaned, featured, supervised, train, val, test)
- output_preprocessing/reports/ : Chứa báo cáo định lượng (quality_report.csv, station_summary.csv, json)
- output_preprocessing/figures/ : Chứa TOÀN BỘ 8 biểu đồ đánh giá chất lượng chuẩn báo cáo khoa học (DPI cao)
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

# Cấu hình thẩm mỹ cho đồ thị chuẩn báo cáo
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

NUMERIC_COLS = ["UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr"]


# =============================================================================
# 1. TÌM KIẾM VÀ ĐỌC FILE CSV
# =============================================================================
def find_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {data_dir}")

    all_csvs = sorted(data_dir.rglob("*.csv"))
    exclude_keywords = ["report", "summary", "cleaned", "featured", "supervised", "train", "validation", "test", "merged_with_features"]

    valid_csvs = [p for p in all_csvs if not any(kw in p.name.lower() for kw in exclude_keywords)]
    return valid_csvs


# =============================================================================
# 2. CHUẨN HÓA THỜI GIAN VÀ CỘT DỮ LIỆU
# =============================================================================
def parse_time_robust(df: pd.DataFrame) -> pd.Series:
    time_candidates = ["TIME", "Time", "ThoiGian", "THOIGIAN", "Gio", "GIO"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    
    if not time_col:
        raise ValueError(f"Không tìm thấy cột thời gian trong: {df.columns.tolist()}")

    s_time = df[time_col].astype(str).str.strip()
    parsed = pd.to_datetime(s_time, errors="coerce", format="mixed")
    if parsed.notna().mean() >= 0.8:
        return parsed

    date_candidates = ["DATE", "Date", "Ngay", "NGAY", "Ngày"]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if date_col:
        s_date = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        s_time_fixed = s_time.str.replace(r"\s+", "", regex=True)
        s_time_fixed = s_time_fixed.where(s_time_fixed.str.count(":") == 2, "00:" + s_time_fixed)
        return s_date + pd.to_timedelta(s_time_fixed, errors="coerce")

    return parsed


def load_and_standardize_single_file(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, low_memory=False, encoding="latin1")

    df.columns = [str(c).strip() for c in df.columns]
    df["TIME"] = parse_time_robust(df)
    df = df.dropna(subset=["TIME"]).copy()

    for c in NUMERIC_COLS:
        if c in df.columns:
            s = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")

    if "MaDiemDo" in df.columns:
        df["MaDiemDo"] = pd.to_numeric(df["MaDiemDo"], errors="coerce")
        df = df.dropna(subset=["MaDiemDo"]).copy()
        df["MaDiemDo"] = df["MaDiemDo"].astype(int)

    if "TenDiemDo" in df.columns:
        df["TenDiemDo"] = df["TenDiemDo"].astype(str).str.strip()

    df["source_file"] = file_path.name
    return df


# =============================================================================
# 3. BÁO CÁO CHẤT LƯỢNG DỮ LIỆU
# =============================================================================
def audit_data_quality(df: pd.DataFrame, reports_dir: Path) -> dict:
    total_rows = len(df)
    unique_points = df["MaDiemDo"].nunique() if "MaDiemDo" in df.columns else 0
    t_min = df["TIME"].min()
    t_max = df["TIME"].max()

    dup_keys = ["TIME", "MaDiemDo"] if "MaDiemDo" in df.columns else ["TIME"]
    num_duplicates = int(df.duplicated(subset=dup_keys).sum())

    missing_stats = df.isna().sum()
    missing_pct = (missing_stats / total_rows * 100).round(2)

    is_aligned_30m = (df["TIME"].dt.minute % 30 == 0) & (df["TIME"].dt.second == 0)
    misaligned_count = int((~is_aligned_30m).sum())

    outliers = {}
    if "P_kW" in df.columns:
        outliers["P_kW_negative"] = int((df["P_kW"] < 0).sum())
        outliers["P_kW_excessive"] = int((df["P_kW"] > 30000).sum())

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
        "duplicate_rows": num_duplicates,
        "misaligned_30m_rows": misaligned_count,
        "outliers": outliers
    }


# =============================================================================
# 4. TẠO LƯỚI THỜI GIAN 30 PHÚT VÀ NỘI SUY CÓ KIỂM SOÁT
# =============================================================================
def resample_and_impute_30m_grid(df: pd.DataFrame) -> pd.DataFrame:
    df["TIME_30m"] = df["TIME"].dt.round("30min")
    df_sorted = df.sort_values(by=["MaDiemDo", "TIME_30m", "TIME"]).reset_index(drop=True)
    df_dedup = df_sorted.drop_duplicates(subset=["MaDiemDo", "TIME_30m"], keep="first").copy()

    resampled_list = []
    for point_id, group in df_dedup.groupby("MaDiemDo"):
        # Xóa cột TIME thô trước khi đưa TIME_30m lên làm index để tránh trùng tên khi reset_index
        if "TIME" in group.columns:
            group = group.drop(columns=["TIME"])
        group = group.set_index("TIME_30m").sort_index()

        t_start = group.index.min()
        t_end = group.index.max()
        full_timeline = pd.date_range(start=t_start, end=t_end, freq="30min", name="TIME")

        group_reindexed = group.reindex(full_timeline)
        group_reindexed["MaDiemDo"] = point_id
        if "TenDiemDo" in group.columns:
            group_reindexed["TenDiemDo"] = group["TenDiemDo"].dropna().iloc[0] if len(group["TenDiemDo"].dropna()) > 0 else f"Point_{point_id}"
        if "MBA" in group.columns:
            group_reindexed["MBA"] = group["MBA"].dropna().iloc[0] if len(group["MBA"].dropna()) > 0 else "Unknown"

        was_missing = group_reindexed["P_kW"].isna()
        present_nums = [c for c in NUMERIC_COLS if c in group_reindexed.columns]

        # Chỉ nội suy tối đa 2 bước liên tiếp (1 tiếng)
        imputed_vals = group_reindexed[present_nums].interpolate(method="time", limit=2, limit_direction="forward")
        is_imputed = was_missing & imputed_vals["P_kW"].notna()
        group_reindexed[present_nums] = imputed_vals
        group_reindexed["is_imputed"] = is_imputed.astype(int)

        resampled_list.append(group_reindexed.reset_index())

    return pd.concat(resampled_list, ignore_index=True)


# =============================================================================
# 5. TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING)
# =============================================================================
def engineer_features(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()
    eps = 1e-6

    # 3 pha
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
        p_prev = grp["P_kW"].shift(1)
        grp["rolling_mean_3"] = p_prev.rolling(3, min_periods=1).mean()
        grp["rolling_mean_6"] = p_prev.rolling(6, min_periods=1).mean()
        grp["rolling_mean_48"] = p_prev.rolling(48, min_periods=1).mean()
        feat_groups.append(grp)

    return pd.concat(feat_groups, ignore_index=True)


# =============================================================================
# 6. TỔNG HỢP DANH SÁCH TRẠM
# =============================================================================
def generate_station_summary(df_clean: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    summary = []
    for pid, grp in df_clean.groupby("MaDiemDo"):
        name = grp["TenDiemDo"].iloc[0] if "TenDiemDo" in grp.columns else f"Point_{pid}"
        p_mean = grp["P_kW"].mean()
        p_max = grp["P_kW"].max()
        missing_rate = grp["P_kW"].isna().mean() * 100

        summary.append({
            "MaDiemDo": pid,
            "TenDiemDo": name,
            "total_records": len(grp),
            "P_mean_kW": round(float(p_mean), 3) if pd.notna(p_mean) else None,
            "P_max_kW": round(float(p_max), 3) if pd.notna(p_max) else None,
            "missing_rate_pct": round(float(missing_rate), 2)
        })

    df_summary = pd.DataFrame(summary).sort_values(by="P_mean_kW", ascending=False).reset_index(drop=True)
    df_summary.to_csv(reports_dir / "station_summary.csv", index=False, encoding="utf-8-sig")
    return df_summary


# =============================================================================
# 7. TẠO TẬP DỮ LIỆU CÓ GIÁM SÁT & CHIA TẬP TRAIN / VAL / TEST
# =============================================================================
def build_supervised_and_split(df_featured: pd.DataFrame, target_point: int, threshold_kw: float = 0.5):
    df_target = df_featured[df_featured["MaDiemDo"] == target_point].sort_values("TIME").copy()

    # Nhãn mục tiêu P[t+1]
    df_target["target_P_kW"] = df_target["P_kW"].shift(-1)
    df_supervised = df_target.dropna(subset=["target_P_kW"]).copy()

    # Độ lệch và nhãn xu hướng
    df_supervised["target_delta_P"] = df_supervised["target_P_kW"] - df_supervised["P_kW"]
    conditions = [
        df_supervised["target_delta_P"] > threshold_kw,
        df_supervised["target_delta_P"] < -threshold_kw
    ]
    df_supervised["target_direction"] = np.select(conditions, ["TĂNG", "GIẢM"], default="KHÔNG ĐỔI")

    # Chia thời gian: Train 70%, Val 15%, Test 15%
    n = len(df_supervised)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    train_df = df_supervised.iloc[:train_idx].copy()
    val_df = df_supervised.iloc[train_idx:val_idx].copy()
    test_df = df_supervised.iloc[val_idx:].copy()

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

    return df_supervised, train_df, val_df, test_df, split_info


# =============================================================================
# 8. BỘ 8 BIỂU ĐỒ ĐÁNH GIÁ CHẤT LƯỢNG ĐÀNG HOÀNG (FIGURES CHUYÊN BIỆT)
# =============================================================================
def generate_evaluation_figures(df_raw: pd.DataFrame, df_clean: pd.DataFrame, df_supervised: pd.DataFrame, target_point: int, figures_dir: Path):
    """
    Sinh 8 biểu đồ đánh giá chuẩn bài báo khoa học, phân giải cao (200 DPI).
    Lưu riêng vào thư mục figures/.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    clean_target = df_clean[df_clean["MaDiemDo"] == target_point].set_index("TIME").sort_index()
    raw_target = df_raw[df_raw["MaDiemDo"] == target_point].set_index("TIME").sort_index()

    # --- ĐỒ THỊ 1: Chuỗi công suất & Phục hồi dữ liệu nội suy (Timeline & Imputation Check) ---
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    # Toàn cảnh 1 tháng
    sample_month = clean_target.iloc[:1440]
    ax1a.plot(sample_month.index, sample_month["P_kW"], color="#1f77b4", lw=1.2, label=f"Trạm {target_point} (30 ngày)")
    ax1a.set_title("1A. Toàn cảnh chuỗi phụ tải P (kW) sau làm sạch (Khung 30 ngày)", fontsize=13, fontweight="bold")
    ax1a.set_ylabel("Công suất P (kW)")
    ax1a.legend(loc="upper right")

    # Zoom cận cảnh 7 ngày: Đối chiếu Gốc vs Sau nội suy
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

    # --- ĐỒ THỊ 2: Tỷ lệ thiếu dữ liệu & Mức độ hoàn thiện ---
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    missing_pct = (df_raw[NUMERIC_COLS].isna().mean() * 100).sort_values(ascending=True)
    bars = ax2.barh(missing_pct.index, missing_pct.values, color=sns.color_palette("Blues_r", len(missing_pct)))
    ax2.set_title("2. Đánh giá tỷ lệ khuyết dữ liệu gốc theo từng biến số (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Tỷ lệ thiếu (%)")
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va="center", fontsize=10, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(figures_dir / "02_data_completeness_missing_matrix.png", dpi=200)
    plt.close(fig2)

    # --- ĐỒ THỊ 3: Phân phối công suất P_kW & Kiểm định loại trừ ngoại lai ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))
    p_data = clean_target["P_kW"].dropna()
    sns.histplot(p_data, bins=60, kde=True, ax=axes3[0], color="#1f77b4", edgecolor="black", alpha=0.6)
    axes3[0].axvline(p_data.mean(), color="red", linestyle="--", lw=2, label=f"Mean = {p_data.mean():.2f} kW")
    axes3[0].axvline(p_data.median(), color="green", linestyle="-.", lw=2, label=f"Median = {p_data.median():.2f} kW")
    axes3[0].set_title("Phân phối xác suất (Histogram & KDE) của P_kW", fontsize=13, fontweight="bold")
    axes3[0].set_xlabel("Công suất P (kW)")
    axes3[0].legend()

    sns.boxplot(y=p_data, ax=axes3[1], color="#98df8a", flierprops={"marker": "x", "markersize": 3, "alpha": 0.5})
    axes3[1].set_title("Biểu đồ Boxplot: Khoảng tứ phân vị & Ngoại lai", fontsize=13, fontweight="bold")
    axes3[1].set_ylabel("Công suất P (kW)")
    fig3.suptitle("3. Kiểm định phân phối và giới hạn ngoại lai công suất P_kW", fontsize=15, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(figures_dir / "03_power_distribution_and_anomaly.png", dpi=200)
    plt.close(fig3)

    # --- ĐỒ THỊ 4: Biểu đồ nhiệt tương quan giữa các đặc trưng (Correlation Heatmap) ---
    fig4, ax4 = plt.subplots(figsize=(11, 8))
    corr_cols = [c for c in ["P_kW", "target_P_kW", "lag_1", "lag_2", "lag_48", "rolling_mean_3", "U_avg", "I_avg", "S_kVA", "CosF"] if c in df_supervised.columns]
    corr_mat = df_supervised[corr_cols].corr()
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="vlag", ax=ax4, cbar_kws={"label": "Hệ số tương quan Pearson"})
    ax4.set_title("4. Ma trận tương quan đặc trưng với biến mục tiêu (target_P_kW)", fontsize=14, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig(figures_dir / "04_correlation_heatmap_features.png", dpi=200)
    plt.close(fig4)

    # --- ĐỒ THỊ 5: Đánh giá biến mục tiêu (Độ biến thiên Delta_P & Cân bằng nhãn xu hướng) ---
    fig5, axes5 = plt.subplots(1, 2, figsize=(15, 6))
    delta_p = df_supervised["target_delta_P"].dropna()
    sns.histplot(delta_p, bins=70, kde=True, ax=axes5[0], color="#9467bd", edgecolor="black")
    axes5[0].axvline(0, color="red", linestyle="--", lw=1.5)
    axes5[0].set_title("Phân phối mức biến thiên Delta P = P[t+1] - P[t] (kW)", fontsize=13, fontweight="bold")
    axes5[0].set_xlabel("Delta P (kW)")

    dir_counts = df_supervised["target_direction"].value_counts()
    colors = {"TĂNG": "#2ca02c", "GIẢM": "#d62728", "KHÔNG ĐỔI": "#7f7f7f"}
    bar_colors = [colors.get(x, "#1f77b4") for x in dir_counts.index]
    bars5 = axes5[1].bar(dir_counts.index, dir_counts.values, color=bar_colors, edgecolor="black")
    axes5[1].set_title("Tỷ lệ nhãn xu hướng (TĂNG / GIẢM / KHÔNG ĐỔI)", fontsize=13, fontweight="bold")
    axes5[1].set_ylabel("Số lượng mẫu")
    total_samples = len(df_supervised)
    for b in bars5:
        h = b.get_height()
        pct = (h / total_samples) * 100
        axes5[1].text(b.get_x() + b.get_width()/2, h + total_samples*0.01, f"{h:,}\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    fig5.suptitle("5. Kiểm định biến mục tiêu dự báo (Target Verification)", fontsize=15, fontweight="bold")
    fig5.tight_layout()
    fig5.savefig(figures_dir / "05_target_evaluation_and_direction.png", dpi=200)
    plt.close(fig5)

    # --- ĐỒ THỊ 6: Scatter Plot P[t] vs P[t+1] (Hồi quy trực quan) ---
    fig6, ax6 = plt.subplots(figsize=(8, 8))
    # Lấy mẫu ngẫu nhiên 3,000 điểm để scatter hiển thị thanh thoát
    sample_scatter = df_supervised.sample(n=min(3000, len(df_supervised)), random_state=42)
    ax6.scatter(sample_scatter["P_kW"], sample_scatter["target_P_kW"], alpha=0.3, color="#1f77b4", edgecolors="none", s=25)
    min_val = min(sample_scatter["P_kW"].min(), sample_scatter["target_P_kW"].min())
    max_val = max(sample_scatter["P_kW"].max(), sample_scatter["target_P_kW"].max())
    ax6.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Đường lý tưởng (y = x)")
    r_val = df_supervised[["P_kW", "target_P_kW"]].corr().iloc[0, 1]
    ax6.set_title(f"6. Tương quan tuyến tính P[t] vs P[t+1] (Pearson r = {r_val:.4f})", fontsize=14, fontweight="bold")
    ax6.set_xlabel("Công suất thời điểm hiện tại P[t] (kW)")
    ax6.set_ylabel("Công suất sau 30 phút P[t+1] (kW)")
    ax6.legend(loc="upper left")
    fig6.tight_layout()
    fig6.savefig(figures_dir / "06_scatter_lag_vs_target.png", dpi=200)
    plt.close(fig6)

    # --- ĐỒ THỊ 7: Tính quy luật mùa vụ theo Giờ & Ngày trong tuần (Seasonality Profile) ---
    fig7, axes7 = plt.subplots(1, 2, figsize=(16, 6))
    clean_target_feat = df_clean[df_clean["MaDiemDo"] == target_point].copy()
    clean_target_feat["hour"] = clean_target_feat["TIME"].dt.hour
    clean_target_feat["dow"] = clean_target_feat["TIME"].dt.dayofweek

    hourly_mean = clean_target_feat.groupby("hour")["P_kW"].mean()
    hourly_std = clean_target_feat.groupby("hour")["P_kW"].std()
    axes7[0].plot(hourly_mean.index, hourly_mean.values, marker="o", color="#ff7f0e", lw=2.5, label="Trung bình")
    axes7[0].fill_between(hourly_mean.index, hourly_mean.values - hourly_std.values, hourly_mean.values + hourly_std.values, color="#ff7f0e", alpha=0.2, label="±1 Std Dev")
    axes7[0].set_title("Phụ tải trung bình theo 24 giờ trong ngày (Hourly Profile)", fontsize=13, fontweight="bold")
    axes7[0].set_xlabel("Giờ trong ngày (0 - 23)")
    axes7[0].set_ylabel("P_kW trung bình")
    axes7[0].set_xticks(range(0, 24))
    axes7[0].legend()

    dow_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
    dow_mean = clean_target_feat.groupby("dow")["P_kW"].mean()
    axes7[1].bar([dow_names[i] for i in dow_mean.index], dow_mean.values, color=sns.color_palette("Set2", 7), edgecolor="black")
    axes7[1].set_title("Phụ tải trung bình theo ngày trong tuần (Weekly Profile)", fontsize=13, fontweight="bold")
    axes7[1].set_ylabel("P_kW trung bình")
    fig7.suptitle("7. Kiểm định tính mùa vụ phụ tải điện (Seasonality Evaluation)", fontsize=15, fontweight="bold")
    fig7.tight_layout()
    fig7.savefig(figures_dir / "07_daily_and_weekly_seasonality.png", dpi=200)
    plt.close(fig7)

    # --- ĐỒ THỊ 8: Trục thời gian phân chia tập dữ liệu (Train - Validation - Test) ---
    fig8, ax8 = plt.subplots(figsize=(15, 4))
    n = len(df_supervised)
    t_idx = int(n * 0.70)
    v_idx = int(n * 0.85)

    df_sup_sorted = df_supervised.sort_values("TIME").reset_index(drop=True)
    ax8.plot(df_sup_sorted.iloc[:t_idx]["TIME"], df_sup_sorted.iloc[:t_idx]["P_kW"], color="#1f77b4", label="Train (70%)", lw=1)
    ax8.plot(df_sup_sorted.iloc[t_idx:v_idx]["TIME"], df_sup_sorted.iloc[t_idx:v_idx]["P_kW"], color="#ff7f0e", label="Validation (15%)", lw=1)
    ax8.plot(df_sup_sorted.iloc[v_idx:]["TIME"], df_sup_sorted.iloc[v_idx:]["P_kW"], color="#2ca02c", label="Test (15%)", lw=1)
    ax8.set_title("8. Phân chia tập dữ liệu tuần tự theo trục thời gian (Không rò rỉ dữ liệu)", fontsize=14, fontweight="bold")
    ax8.set_xlabel("Thời gian")
    ax8.set_ylabel("P (kW)")
    ax8.legend(loc="upper right")
    fig8.tight_layout()
    fig8.savefig(figures_dir / "08_data_split_timeline.png", dpi=200)
    plt.close(fig8)


# =============================================================================
# HÀM THỰC THI PIPELINE VỚI CẤU TRÚC THƯ MỤC TÁCH BẠCH
# =============================================================================
def run_pipeline(
    data_dir_str: str,
    output_dir_str: str = "output_preprocessing",
    target_point: int = None,
    threshold_kw: float = 0.5
):
    base_out = Path(output_dir_str)
    data_out = base_out / "data"
    reports_out = base_out / "reports"
    figures_out = base_out / "figures"

    # Tạo các thư mục riêng biệt
    for d in [data_out, reports_out, figures_out]:
        d.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir_str)

    print("=" * 75)
    print("PIPELINE TIỀN XỬ LÝ PHỤ TẢI ĐIỆN 30 PHÚT - CẤU TRÚC ĐẦU RA TÁCH BẠCH")
    print(f"📁 Dữ liệu đầu vào:      {data_dir.resolve()}")
    print(f"💾 Thư mục dữ liệu sạch: {data_out.resolve()}")
    print(f"📑 Thư mục báo cáo CSV:  {reports_out.resolve()}")
    print(f"📊 Thư mục biểu đồ (DPI cao): {figures_out.resolve()}")
    print("=" * 75)

    # 1. Quét CSV
    csv_files = find_csv_files(data_dir)
    print(f"\n[1/6] Tìm thấy {len(csv_files)} file CSV hợp lệ. Đang đọc...")
    dfs = [load_and_standardize_single_file(f) for f in csv_files]
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"      -> Tổng số dòng dữ liệu thô: {len(df_raw):,}")

    # 2. Kiểm tra chất lượng -> lưu vào reports/
    print("\n[2/6] Xuất báo cáo chất lượng: reports/quality_report.csv...")
    audit_summary = audit_data_quality(df_raw, reports_out)

    # 3. Làm sạch, lưới 30m, xuất cleaned_long & cleaned_wide -> data/
    print("\n[3/6] Đồng bộ lưới thời gian 30 phút, lưu vào data/cleaned_long.csv & cleaned_wide.csv...")
    df_clean_long = resample_and_impute_30m_grid(df_raw)
    df_clean_long.to_csv(data_out / "cleaned_long.csv", index=False, encoding="utf-8-sig")

    df_wide = df_clean_long.pivot_table(index="TIME", columns="MaDiemDo", values=["P_kW", "Q_kVAr"], aggfunc="first")
    df_wide.columns = [f"{var}_{pt}" for (var, pt) in df_wide.columns]
    df_wide.reset_index().to_csv(data_out / "cleaned_wide.csv", index=False, encoding="utf-8-sig")

    # 4. Trích xuất đặc trưng & Chọn trạm lớn
    print("\n[4/6] Trích xuất đặc trưng vật lý & chuỗi thời gian, lưu data/featured_data.csv...")
    df_featured = engineer_features(df_clean_long)
    df_featured.to_csv(data_out / "featured_data.csv", index=False, encoding="utf-8-sig")

    df_station_summary = generate_station_summary(df_clean_long, reports_out)
    if target_point is None:
        target_point = int(df_station_summary.iloc[0]["MaDiemDo"])
    print(f"      -> Trạm mục tiêu chính: MaDiemDo = {target_point} ({df_station_summary.iloc[0]['TenDiemDo']})")

    # 5. Tạo dữ liệu có giám sát & Chia tập -> data/
    print("\n[5/6] Tạo biến mục tiêu sau 30 phút, chia Train (70%) - Val (15%) - Test (15%)...")
    df_supervised, train_df, val_df, test_df, split_info = build_supervised_and_split(df_featured, target_point, threshold_kw)
    df_supervised.to_csv(data_out / "supervised_data.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(data_out / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(data_out / "validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(data_out / "test.csv", index=False, encoding="utf-8-sig")
    print(f"      -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # 6. Vẽ bộ 8 biểu đồ đánh giá chuẩn bài báo khoa học -> figures/
    print("\n[6/6] Tự động sinh bộ 8 biểu đồ đánh giá chất lượng cao vào figures/...")
    generate_evaluation_figures(df_raw, df_clean_long, df_supervised, target_point, figures_out)
    print("      -> Đã xuất đủ 8 biểu đồ đánh giá chuẩn báo cáo khoa học (DPI 200).")

    # Lưu metadata tổng kết vào reports/
    summary_final = {
        "timestamp": datetime.now().isoformat(),
        "total_files_read": len(csv_files),
        "raw_rows": len(df_raw),
        "clean_rows": len(df_clean_long),
        "selected_target_point": target_point,
        "split_info": split_info,
        "paths": {
            "clean_data_dir": str(data_out.resolve()),
            "reports_dir": str(reports_out.resolve()),
            "figures_dir": str(figures_out.resolve())
        }
    }
    with open(reports_out / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_final, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("✅ TOÀN BỘ DỮ LIỆU SẠCH, BÁO CÁO VÀ BIỂU ĐỒ ĐÃ ĐƯỢC LƯU RIÊNG BIỆT THÀNH CÔNG!")
    print("=" * 75)
    return summary_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline tiền xử lý phụ tải điện 30 phút")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output_preprocessing")
    parser.add_argument("--target_point", type=int, default=None)
    parser.add_argument("--threshold_kw", type=float, default=0.5)
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
