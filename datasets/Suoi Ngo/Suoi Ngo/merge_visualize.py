import os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== INPUTS ======
INPUTS = [
    r"C:\Users\DELL\Desktop\Tam_TLU\datasets\Suoi Ngo\Suoi Ngo\2024\output_suoingo\SUOINGO_2024_T1_to_T9_merged.csv",
    r"C:\Users\DELL\Desktop\Tam_TLU\datasets\Suoi Ngo\Suoi Ngo\2025\output_suoingo\SUOINGO_2025_T1_to_T9_merged.csv",
]
OUT_DIR = Path(r"C:\Users\DELL\Desktop\Tam_TLU\datasets\Suoi Ngo\Suoi Ngo\output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
    for c in ["UA","UB","UC","IA","IB","IC","CosF","P_kW","Q_kVAr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["__src_merged"] = path.name
    return df

# ====== 1) MERGE ======
dfs = [read_merged(Path(p)) for p in INPUTS]
df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=["TIME"]).sort_values("TIME").reset_index(drop=True)

# dedup (nếu có)
dedup_keys = [c for c in ["MaDiemDo","TIME","UA","UB","UC","IA","IB","IC","CosF","P_kW","Q_kVAr"] if c in df.columns]
df = df.drop_duplicates(subset=dedup_keys, keep="first")

# ====== 2) FEATURE ENGINEERING (3 pha) ======
eps = 1e-6
U_EPS = 10.0   # V
I_EPS = 0.5    # A

df["U_avg_V"] = df[["UA","UB","UC"]].mean(axis=1)
df["I_avg_A"] = df[["IA","IB","IC"]].mean(axis=1)

u_dev = (df[["UA","UB","UC"]].sub(df["U_avg_V"], axis=0)).abs()
i_dev = (df[["IA","IB","IC"]].sub(df["I_avg_A"], axis=0)).abs()

df["U_unbalance_pct"] = np.where(df["U_avg_V"].abs() >= U_EPS,
                                 u_dev.max(axis=1)/(df["U_avg_V"].abs()+eps)*100,
                                 np.nan)
df["I_unbalance_pct"] = np.where(df["I_avg_A"].abs() >= I_EPS,
                                 i_dev.max(axis=1)/(df["I_avg_A"].abs()+eps)*100,
                                 np.nan)

df["S_kVA"] = np.sqrt(df["P_kW"]**2 + df["Q_kVAr"]**2)
df["pf_est"] = df["P_kW"]/(df["S_kVA"]+eps)
df["pf_est_abs"] = df["pf_est"].abs()

# clip chỉ để vẽ
p01, p99 = df["P_kW"].quantile(0.01), df["P_kW"].quantile(0.99)
df["P_kW_clip"] = df["P_kW"].clip(p01, p99)

out_csv = OUT_DIR / "SUOINGO_merged_with_features.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

# ====== 3) VISUALIZATION (các đồ thị chính) ======
def savefig_fast(fig, path: Path, dpi=140):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

dfq = df.copy()
dfq["date"] = dfq["TIME"].dt.date
dfq["dow"]  = dfq["TIME"].dt.dayofweek
dfq["hour"] = dfq["TIME"].dt.hour
df_ts = dfq.set_index("TIME")

# (A) P theo giờ (clipped) theo từng điểm đo
hourlyP = (df_ts.groupby("TenDiemDo").resample("1h")["P_kW_clip"].mean().reset_index())
pivot_hourlyP = hourlyP.pivot(index="TIME", columns="TenDiemDo", values="P_kW_clip")
fig, ax = plt.subplots()
pivot_hourlyP.plot(ax=ax)
ax.set_title("P_kW trung bình theo giờ (clipped p01–p99)")
ax.set_xlabel("Thời gian"); ax.set_ylabel("kW")
savefig_fast(fig, OUT_DIR/"01_hourly_PkW_clipped_all_points.png")

# (B) Năng lượng ngày (kWh) import/export từ P (30 phút mean × 0.5h)
half_hour = (df_ts.groupby("TenDiemDo").resample("30min")["P_kW_clip"].mean().reset_index())
half_hour["date"] = half_hour["TIME"].dt.date
half_hour["P_pos"] = half_hour["P_kW_clip"].clip(lower=0)
half_hour["P_neg"] = (-half_hour["P_kW_clip"].clip(upper=0))
daily_ie = (half_hour.groupby(["TenDiemDo","date"])
            .agg(import_kWh=("P_pos", lambda s: float(s.sum()*0.5)),
                 export_kWh=("P_neg", lambda s: float(s.sum()*0.5)))
            .reset_index())
daily_ie.to_csv(OUT_DIR/"daily_import_export_kWh.csv", index=False, encoding="utf-8-sig")

pivot_import = daily_ie.pivot(index="date", columns="TenDiemDo", values="import_kWh")
fig, ax = plt.subplots()
pivot_import.plot(ax=ax)
ax.set_title("Điện năng nhập theo ngày (kWh) (P>0, clipped)")
ax.set_xlabel("Ngày"); ax.set_ylabel("kWh nhập")
savefig_fast(fig, OUT_DIR/"08_daily_import_kWh_all_points.png")

pivot_export = daily_ie.pivot(index="date", columns="TenDiemDo", values="export_kWh")
fig, ax = plt.subplots()
pivot_export.plot(ax=ax)
ax.set_title("Điện năng phát/ngược theo ngày (kWh) (P<0, clipped)")
ax.set_xlabel("Ngày"); ax.set_ylabel("kWh phát/ngược")
savefig_fast(fig, OUT_DIR/"09_daily_export_kWh_all_points.png")

# (C) Heatmap phụ tải: Thứ × Giờ (overall mean)
hm = (dfq.dropna(subset=["P_kW_clip"]).groupby(["dow","hour"])["P_kW_clip"].mean().reset_index())
mat = np.full((7,24), np.nan)
for _, r in hm.iterrows():
    mat[int(r["dow"]), int(r["hour"])] = r["P_kW_clip"]
fig, ax = plt.subplots()
im = ax.imshow(mat, aspect="auto")
ax.set_title("Heatmap P_kW (clipped) theo Thứ (0=Mon) × Giờ (overall)")
ax.set_xlabel("Giờ (0-23)"); ax.set_ylabel("Thứ (0=Mon)")
fig.colorbar(im, ax=ax)
savefig_fast(fig, OUT_DIR/"22_heatmap_P_by_dow_hour_overall.png")

# # ====== 4) ZIP toàn bộ output ======
# zip_path = Path("/mnt/data/SUOINGO_scientific_viz_outputs.zip")
# with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
#     for root, _, files in os.walk(OUT_DIR):
#         for f in files:
#             fp = Path(root) / f
#             z.write(fp, arcname=str(fp.relative_to(OUT_DIR)))

# print("Saved:", out_csv)
# print("ZIP:", zip_path)
