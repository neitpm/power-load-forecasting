# merge_and_visualize_suoingo.py
# ✅ Gộp nhiều CSV (T1..T9) + trực quan hóa (matplotlib)
# Chạy được trên local / Colab (chỉ cần pandas, matplotlib)

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1) CONFIG
# =========================
# Nếu bạn đang dùng Colab, có thể đặt DATA_DIR là thư mục trên Drive.
DATA_DIR = Path(".")  # đổi thành Path("/content/drive/MyDrive/...") nếu cần

# Nếu muốn chỉ định chính xác danh sách file (khuyên dùng nếu bạn đã biết tên):
FILES = [
    "T1-2024-T2-SUOINGO.csv",
    "T2-2024-T3-SUOINGO.csv",
    "T3-2024-T4-SUOINGO.csv",
    "T4-2024-T5-SUOINGO.csv",
    "T5-2024-T6-SUOINGO.csv",
    "T6-2024-T7-SUOINGO.csv",
    "T7-2024-T8-SUOINGO.csv",
    "T8-2024-T9-SUOINGO.csv",
    "T9-2024-T10-SUOINGO.csv",
    "T10-2024-T11-SUOINGO.csv",
    "T11-2024-T12-SUOINGO.csv",
    "T12-2024-T12-SUOINGO.csv",
]

OUT_DIR = Path("output_suoingo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MERGED_CSV = OUT_DIR / "SUOINGO_2024_T1_to_T9_merged.csv"


# =========================
# 2) LOAD + MERGE
# =========================
def read_csv_safe(path: Path) -> pd.DataFrame:
    """
    Đọc CSV utf-8-sig, delimiter mặc định ','.
    Nếu dữ liệu bạn có thể dùng ';' thì đổi sep=';'.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", sep=",")
    df["__source_file"] = path.name
    return df


def merge_all(files):
    dfs = []
    for fn in files:
        p = DATA_DIR / fn
        if not p.exists():
            raise FileNotFoundError(f"Không thấy file: {p.resolve()}")
        dfs.append(read_csv_safe(p))

    df_all = pd.concat(dfs, ignore_index=True)

    # Parse TIME
    df_all["TIME"] = pd.to_datetime(df_all["TIME"], errors="coerce")

    # Ép numeric các cột đo
    num_cols = ["UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr"]
    for c in num_cols:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # Sort & drop invalid TIME
    df_all = df_all.dropna(subset=["TIME"]).sort_values("TIME").reset_index(drop=True)

    # Deduplicate (nếu bạn muốn khắt khe hơn/nhẹ hơn thì chỉnh subset)
    dedup_keys = ["MaDiemDo", "TIME", "UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr"]
    df_all = df_all.drop_duplicates(subset=[c for c in dedup_keys if c in df_all.columns], keep="first")

    return df_all


df_all = merge_all(FILES)

print("Merged rows:", len(df_all))
print("Time range:", df_all["TIME"].min(), "->", df_all["TIME"].max())
print("Unique TenDiemDo:", df_all["TenDiemDo"].nunique())

# Lưu file gộp
df_all.to_csv(OUT_MERGED_CSV, index=False, encoding="utf-8-sig")
print("Saved:", OUT_MERGED_CSV.resolve())


# =========================
# 3) OUTLIER CLIP (chỉ để trực quan hóa)
# =========================
p01 = df_all["P_kW"].quantile(0.01)
p99 = df_all["P_kW"].quantile(0.99)
print("P_kW p01/p99 =", p01, p99)

df_vis = df_all.copy()
df_vis["P_kW_clip"] = df_vis["P_kW"].clip(lower=p01, upper=p99)


# =========================
# 4) VISUALIZATION
# =========================
def savefig(path: Path, dpi=160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print("Saved plot:", path.resolve())


# 4.1 Hourly mean P_kW per point (clipped)
df_p = df_vis[["TIME", "TenDiemDo", "P_kW_clip"]].dropna(subset=["P_kW_clip"]).copy()
df_p = df_p.set_index("TIME")

hourly = (
    df_p.groupby("TenDiemDo")
        .resample("1h")["P_kW_clip"].mean()
        .reset_index()
)

pivot_hourly = hourly.pivot(index="TIME", columns="TenDiemDo", values="P_kW_clip")

plt.figure()
pivot_hourly.plot(ax=plt.gca())
plt.title("P_kW trung bình theo giờ (clipped p01–p99)")
plt.xlabel("Thời gian")
plt.ylabel("P_kW mean (clipped)")
savefig(OUT_DIR / "plot_hourly_PkW_clipped.png")


# 4.2 Daily kWh estimate (30min mean × 0.5h) using clipped P_kW
half_hour = (
    df_p.groupby("TenDiemDo")
        .resample("30min")["P_kW_clip"].mean()
        .reset_index()
)
half_hour["date"] = half_hour["TIME"].dt.date

daily_kwh = (
    (half_hour.groupby(["TenDiemDo", "date"])["P_kW_clip"].sum() * 0.5)
    .reset_index(name="kWh_est_clip")
)

pivot_daily = daily_kwh.pivot(index="date", columns="TenDiemDo", values="kWh_est_clip")

plt.figure()
pivot_daily.plot(ax=plt.gca())
plt.title("Điện năng ước tính theo ngày (kWh) (clipped p01–p99)")
plt.xlabel("Ngày")
plt.ylabel("kWh ước tính (clipped)")
savefig(OUT_DIR / "plot_daily_kWh_clipped.png")


# 4.3 Import/export kWh per day (clipped)
half_hour["P_pos"] = half_hour["P_kW_clip"].clip(lower=0)          # nhập
half_hour["P_neg"] = (-half_hour["P_kW_clip"].clip(upper=0))       # phát/ngược (đổi thành dương)

daily_ie = (
    half_hour.groupby(["TenDiemDo", "date"])
    .agg(
        import_kWh=("P_pos", lambda s: float(s.sum() * 0.5)),
        export_kWh=("P_neg", lambda s: float(s.sum() * 0.5)),
    )
    .reset_index()
)

pivot_import = daily_ie.pivot(index="date", columns="TenDiemDo", values="import_kWh")
pivot_export = daily_ie.pivot(index="date", columns="TenDiemDo", values="export_kWh")

plt.figure()
pivot_import.plot(ax=plt.gca())
plt.title("Điện năng nhập (kWh/ngày) (P>0, clipped)")
plt.xlabel("Ngày")
plt.ylabel("import kWh")
savefig(OUT_DIR / "plot_daily_import_kWh.png")

plt.figure()
pivot_export.plot(ax=plt.gca())
plt.title("Điện năng phát/ngược (kWh/ngày) (P<0, clipped)")
plt.xlabel("Ngày")
plt.ylabel("export kWh")
savefig(OUT_DIR / "plot_daily_export_kWh.png")


# 4.4 Histogram P_kW clipped
plt.figure()
df_vis["P_kW_clip"].dropna().plot(kind="hist", bins=60)
plt.title("Phân phối P_kW (clipped p01–p99)")
plt.xlabel("P_kW (clipped)")
plt.ylabel("Tần suất")
savefig(OUT_DIR / "plot_hist_PkW_clipped.png")


# 4.5 Average P_kW by hour of day (overall) - dùng P_kW gốc hoặc clipped tuỳ bạn
tmp = df_vis[["TIME", "P_kW_clip"]].dropna().copy()
tmp["hour"] = tmp["TIME"].dt.hour
avg_by_hour = tmp.groupby("hour")["P_kW_clip"].mean().reindex(range(24))

plt.figure()
avg_by_hour.plot(kind="bar")
plt.title("P_kW trung bình theo giờ trong ngày (overall, clipped)")
plt.xlabel("Giờ (0-23)")
plt.ylabel("P_kW mean (clipped)")
savefig(OUT_DIR / "plot_avg_by_hour.png")


# =========================
# 5) QUICK QA: tìm ngoại lai lớn để kiểm tra
# =========================
thr = 100  # kW
extreme = df_all.loc[df_all["P_kW"].abs() > thr, ["TIME", "TenDiemDo", "P_kW", "UA", "IA", "CosF", "__source_file"]]
print("Extreme rows |P_kW|>100:", len(extreme))
(extreme.sort_values("P_kW", key=lambda s: s.abs(), ascending=False)
        .head(20)
        .to_csv(OUT_DIR / "top20_extreme_PkW.csv", index=False, encoding="utf-8-sig"))

print("Done. Outputs in:", OUT_DIR.resolve())
