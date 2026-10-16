# -*- coding: utf-8 -*-
"""
=============================================================================
KIỂM ĐỊNH THỐNG KÊ Ý NGHĨA VỚI MOVING-BLOCK BOOTSTRAP (M-BB)
BẢN CẢI TIẾN: HỖ TRỢ ĐA KHỐI (48 BƯỚC / 24H & 336 BƯỚC / 7 NGÀY)
VÀ XỬ LÝ CHẶT CHẼ RANH GIỚI CÁC PHÂN ĐOẠN LIÊN TỤC (CONTIGUOUS SEGMENTS)
=============================================================================
Đặc điểm:
1. Xác định 16 phân đoạn liên tục trong tập đánh giá 14,385 mẫu.
2. Khi lấy mẫu khối (Block Sampling), chỉ lấy các block nằm trọn trong cùng một
   phân đoạn liên tục (không lấy block bắc cầu qua khoảng đứt gãy thời gian).
3. Đánh giá đa quy mô:
   - Block = 48 bước (1 ngày - chu kỳ phụ tải ngày).
   - Block = 336 bước (7 ngày - chu kỳ tuần).
4. Tính toán đầy đủ:
   - ΔRMSE và ΔMAE
   - Khoảng tin cậy 95% CI
   - Tỷ lệ mô hình cây tốt hơn Persistence
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def build_contiguous_blocks(segment_ids, block_length):
    """
    Xác định tất cả các điểm bắt đầu hợp lệ sao cho toàn bộ block dài block_length
    nằm trọn trong cùng một phân đoạn thời gian liên tục.
    """
    valid_starts = []
    n = len(segment_ids)
    for i in range(n - block_length + 1):
        if segment_ids[i] == segment_ids[i + block_length - 1]:
            valid_starts.append(i)
    return np.array(valid_starts)


def contiguous_moving_block_bootstrap(
    y_true, y_pred_cand, y_pred_base, segment_ids,
    block_length=48, n_boot=1000, seed=42
):
    np.random.seed(seed)
    n = len(y_true)
    valid_starts = build_contiguous_blocks(segment_ids, block_length)

    if len(valid_starts) == 0:
        raise ValueError(f"Không tìm thấy phân đoạn liên tục nào đủ dài {block_length} bước.")

    num_blocks = int(np.ceil(n / block_length))

    diff_rmse_list = []
    diff_mae_list = []

    rmse_base_orig = np.sqrt(mean_squared_error(y_true, y_pred_base))
    rmse_cand_orig = np.sqrt(mean_squared_error(y_true, y_pred_cand))
    mae_base_orig = mean_absolute_error(y_true, y_pred_base)
    mae_cand_orig = mean_absolute_error(y_true, y_pred_cand)

    for b in range(n_boot):
        chosen_starts = np.random.choice(valid_starts, size=num_blocks, replace=True)
        boot_indices = np.concatenate([np.arange(s, s + block_length) for s in chosen_starts])[:n]

        yt = y_true[boot_indices]
        yp_c = y_pred_cand[boot_indices]
        yp_b = y_pred_base[boot_indices]

        rmse_c = np.sqrt(np.mean((yt - yp_c) ** 2))
        rmse_b = np.sqrt(np.mean((yt - yp_b) ** 2))
        diff_rmse = rmse_b - rmse_c  # > 0 nghĩa là Candidate tốt hơn (giảm RMSE)

        mae_c = np.mean(np.abs(yt - yp_c))
        mae_b = np.mean(np.abs(yt - yp_b))
        diff_mae = mae_b - mae_c

        diff_rmse_list.append(diff_rmse)
        diff_mae_list.append(diff_mae)

    diff_rmse_arr = np.array(diff_rmse_list)
    diff_mae_arr = np.array(diff_mae_list)

    ci_rmse_low, ci_rmse_high = np.percentile(diff_rmse_arr, [2.5, 97.5])
    ci_mae_low, ci_mae_high = np.percentile(diff_mae_arr, [2.5, 97.5])

    return {
        "block_length": block_length,
        "valid_blocks_count": len(valid_starts),
        "rmse_base": rmse_base_orig,
        "rmse_cand": rmse_cand_orig,
        "delta_rmse_mean": np.mean(diff_rmse_arr),
        "ci_rmse_95": (ci_rmse_low, ci_rmse_high),
        "rmse_better_pct": np.mean(diff_rmse_arr > 0) * 100,
        "mae_base": mae_base_orig,
        "mae_cand": mae_cand_orig,
        "delta_mae_mean": np.mean(diff_mae_arr),
        "ci_mae_95": (ci_mae_low, ci_mae_high),
        "mae_better_pct": np.mean(diff_mae_arr > 0) * 100,
        "diff_rmse_samples": diff_rmse_arr,
        "diff_mae_samples": diff_mae_arr
    }


def main():
    base_dir = Path("output_preprocessing")
    data_dir = base_dir / "data"
    reports_dir = base_dir / "reports"
    figures_dir = base_dir / "figures"

    print("=" * 80)
    print("CHẠY KIỂM ĐỊNH CONTIGUOUS MOVING-BLOCK BOOTSTRAP (BLOCK 48 & BLOCK 336)")
    print("=" * 80)

    # 1. Đọc dữ liệu
    test_df = pd.read_csv(data_dir / "test.csv")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")

    test_df["TIME"] = pd.to_datetime(test_df["TIME"])
    test_df = test_df.sort_values("TIME").reset_index(drop=True)

    candidate_features = [
        "UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr",
        "U_avg", "U_max", "U_min", "U_unbalance_pct",
        "I_avg", "I_max", "I_min", "I_unbalance_pct",
        "S_kVA", "lag_1", "lag_2", "lag_3", "lag_48", "lag_336",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_48",
        "hour", "dayofweek", "dayofmonth", "month", "is_weekend",
        "sin_hour", "cos_hour"
    ]
    feature_cols = [c for c in candidate_features if c in train_df.columns]
    target_col = "target_P_kW"

    # Lọc mask chung 14,385 mẫu
    pred_persistence = test_df["P_kW"].values
    pred_s48 = test_df["lag_48"].values
    pred_s336 = test_df["lag_336"].values
    pred_ma3 = test_df["rolling_mean_3"].values
    y_test = test_df[target_col].values

    mask_common = (
        np.isfinite(y_test) &
        np.isfinite(pred_persistence) &
        np.isfinite(pred_s48) &
        np.isfinite(pred_s336) &
        np.isfinite(pred_ma3)
    )

    eval_df = test_df[mask_common].copy().reset_index(drop=True)
    y_eval = y_test[mask_common]
    p_eval = pred_persistence[mask_common]

    # Tính toán chính xác các phân đoạn thời gian liên tục (Contiguous Segments)
    diffs = eval_df["TIME"].diff()
    segment_ids = (diffs != pd.Timedelta(minutes=30)).cumsum().values
    n_segments = len(np.unique(segment_ids))
    print(f"-> Tập đánh giá chuẩn: {len(y_eval):,} mẫu, gồm {n_segments} phân đoạn thời gian liên tục.")

    # Huấn luyện XGBoost & HistGBDT
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(train_df[feature_cols])
    y_train = train_df[target_col].values
    X_val_imp = imputer.transform(val_df[feature_cols])
    y_val = val_df[target_col].values
    X_test_imp = imputer.transform(test_df[feature_cols])

    print("\nHuấn luyện XGBoost Regressor...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, early_stopping_rounds=30
    )
    model_xgb.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], verbose=False)
    pred_xgb = model_xgb.predict(X_test_imp)[mask_common]

    print("Huấn luyện HistGradientBoosting Regressor...")
    model_hgb = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=6, random_state=42, early_stopping=True
    )
    model_hgb.fit(X_train_imp, y_train)
    pred_hgb = model_hgb.predict(X_test_imp)[mask_common]

    # Kiểm định với Block 48 (24h) và Block 336 (7 ngày)
    configs = [
        {"name": "24h (48 bước)", "block": 48},
        {"name": "7 ngày (336 bước)", "block": 336}
    ]

    all_results = {}

    for cfg in configs:
        b_len = cfg["block"]
        b_name = cfg["name"]
        print(f"\n" + "-" * 70)
        print(f"TIẾN HÀNH CONTIGUOUS M-BB CHO KHUNG {b_name}...")
        print("-" * 70)

        res_xgb = contiguous_moving_block_bootstrap(
            y_eval, pred_xgb, p_eval, segment_ids, block_length=b_len, n_boot=1000
        )
        res_hgb = contiguous_moving_block_bootstrap(
            y_eval, pred_hgb, p_eval, segment_ids, block_length=b_len, n_boot=1000
        )

        print(f"XGBoost vs Persistence (Block {b_len}):")
        print(f"  ΔRMSE trung bình: {res_xgb['delta_rmse_mean']:+.2f} kW | 95% CI: [{res_xgb['ci_rmse_95'][0]:.2f}; {res_xgb['ci_rmse_95'][1]:.2f}] kW")
        print(f"  Tỷ lệ thắng RMSE: {res_xgb['rmse_better_pct']:.1f}%")
        print(f"  ΔMAE trung bình:  {res_xgb['delta_mae_mean']:+.2f} kW | 95% CI: [{res_xgb['ci_mae_95'][0]:.2f}; {res_xgb['ci_mae_95'][1]:.2f}] kW")

        print(f"HistGBDT vs Persistence (Block {b_len}):")
        print(f"  ΔRMSE trung bình: {res_hgb['delta_rmse_mean']:+.2f} kW | 95% CI: [{res_hgb['ci_rmse_95'][0]:.2f}; {res_hgb['ci_rmse_95'][1]:.2f}] kW")
        print(f"  Tỷ lệ thắng RMSE: {res_hgb['rmse_better_pct']:.1f}%")
        print(f"  ΔMAE trung bình:  {res_hgb['delta_mae_mean']:+.2f} kW | 95% CI: [{res_hgb['ci_mae_95'][0]:.2f}; {res_hgb['ci_mae_95'][1]:.2f}] kW")

        all_results[f"block_{b_len}"] = {
            "block_length_steps": b_len,
            "block_duration": b_name,
            "valid_blocks_count": res_xgb["valid_blocks_count"],
            "XGBoost_vs_Persistence": {
                "delta_rmse_mean_kW": round(float(res_xgb['delta_rmse_mean']), 2),
                "ci_rmse_95_kW": [round(float(x), 2) for x in res_xgb['ci_rmse_95']],
                "rmse_better_pct": round(float(res_xgb['rmse_better_pct']), 1),
                "delta_mae_mean_kW": round(float(res_xgb['delta_mae_mean']), 2),
                "ci_mae_95_kW": [round(float(x), 2) for x in res_xgb['ci_mae_95']],
                "is_significant_at_005": bool(res_xgb['ci_rmse_95'][0] > 0)
            },
            "HistGBDT_vs_Persistence": {
                "delta_rmse_mean_kW": round(float(res_hgb['delta_rmse_mean']), 2),
                "ci_rmse_95_kW": [round(float(x), 2) for x in res_hgb['ci_rmse_95']],
                "rmse_better_pct": round(float(res_hgb['rmse_better_pct']), 1),
                "delta_mae_mean_kW": round(float(res_hgb['delta_mae_mean']), 2),
                "ci_mae_95_kW": [round(float(x), 2) for x in res_hgb['ci_mae_95']],
                "is_significant_at_005": bool(res_hgb['ci_rmse_95'][0] > 0)
            }
        }

    # Lưu JSON báo cáo tổng hợp
    full_report = {
        "method": "Contiguous Moving-Block Bootstrap (Zero Cross-Gap Leakage, B=1000)",
        "evaluated_samples": len(y_eval),
        "contiguous_segments_count": n_segments,
        "results": all_results
    }
    with open(reports_dir / "moving_block_bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)

    # Vẽ biểu đồ so sánh 2x2: (Block 48 & Block 336) cho XGBoost và HistGBDT
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Hàng 1: Block 48
    r48_xgb = contiguous_moving_block_bootstrap(y_eval, pred_xgb, p_eval, segment_ids, block_length=48, n_boot=1000)
    r48_hgb = contiguous_moving_block_bootstrap(y_eval, pred_hgb, p_eval, segment_ids, block_length=48, n_boot=1000)

    sns.histplot(r48_xgb["diff_rmse_samples"], bins=40, kde=True, ax=axes[0, 0], color="#ff7f0e", alpha=0.6)
    axes[0, 0].axvline(0, color="black", ls="--", lw=1.5, label="Không chênh lệch (0 kW)")
    axes[0, 0].axvline(r48_xgb['ci_rmse_95'][0], color="red", ls=":", lw=1.5, label=f"2.5%: {r48_xgb['ci_rmse_95'][0]:.1f}")
    axes[0, 0].axvline(r48_xgb['ci_rmse_95'][1], color="red", ls=":", lw=1.5, label=f"97.5%: {r48_xgb['ci_rmse_95'][1]:.1f}")
    axes[0, 0].set_title(f"A. XGBoost vs Persistence [Block 48 (24h)]\nΔRMSE: {r48_xgb['delta_rmse_mean']:+.1f} kW | Thắng: {r48_xgb['rmse_better_pct']:.1f}%", fontweight="bold")
    axes[0, 0].set_xlabel("ΔRMSE (Persistence - XGBoost) [kW]")
    axes[0, 0].legend()

    sns.histplot(r48_hgb["diff_rmse_samples"], bins=40, kde=True, ax=axes[0, 1], color="#2ca02c", alpha=0.6)
    axes[0, 1].axvline(0, color="black", ls="--", lw=1.5, label="Không chênh lệch (0 kW)")
    axes[0, 1].axvline(r48_hgb['ci_rmse_95'][0], color="red", ls=":", lw=1.5, label=f"2.5%: {r48_hgb['ci_rmse_95'][0]:.1f}")
    axes[0, 1].axvline(r48_hgb['ci_rmse_95'][1], color="red", ls=":", lw=1.5, label=f"97.5%: {r48_hgb['ci_rmse_95'][1]:.1f}")
    axes[0, 1].set_title(f"B. HistGBDT vs Persistence [Block 48 (24h)]\nΔRMSE: {r48_hgb['delta_rmse_mean']:+.1f} kW | Thắng: {r48_hgb['rmse_better_pct']:.1f}%", fontweight="bold")
    axes[0, 1].set_xlabel("ΔRMSE (Persistence - HistGBDT) [kW]")
    axes[0, 1].legend()

    # Hàng 2: Block 336
    r336_xgb = contiguous_moving_block_bootstrap(y_eval, pred_xgb, p_eval, segment_ids, block_length=336, n_boot=1000)
    r336_hgb = contiguous_moving_block_bootstrap(y_eval, pred_hgb, p_eval, segment_ids, block_length=336, n_boot=1000)

    sns.histplot(r336_xgb["diff_rmse_samples"], bins=40, kde=True, ax=axes[1, 0], color="#ff7f0e", alpha=0.6)
    axes[1, 0].axvline(0, color="black", ls="--", lw=1.5, label="Không chênh lệch (0 kW)")
    axes[1, 0].axvline(r336_xgb['ci_rmse_95'][0], color="red", ls=":", lw=1.5, label=f"2.5%: {r336_xgb['ci_rmse_95'][0]:.1f}")
    axes[1, 0].axvline(r336_xgb['ci_rmse_95'][1], color="red", ls=":", lw=1.5, label=f"97.5%: {r336_xgb['ci_rmse_95'][1]:.1f}")
    axes[1, 0].set_title(f"C. XGBoost vs Persistence [Block 336 (7 ngày)]\nΔRMSE: {r336_xgb['delta_rmse_mean']:+.1f} kW | Thắng: {r336_xgb['rmse_better_pct']:.1f}%", fontweight="bold")
    axes[1, 0].set_xlabel("ΔRMSE (Persistence - XGBoost) [kW]")
    axes[1, 0].legend()

    sns.histplot(r336_hgb["diff_rmse_samples"], bins=40, kde=True, ax=axes[1, 1], color="#2ca02c", alpha=0.6)
    axes[1, 1].axvline(0, color="black", ls="--", lw=1.5, label="Không chênh lệch (0 kW)")
    axes[1, 1].axvline(r336_hgb['ci_rmse_95'][0], color="red", ls=":", lw=1.5, label=f"2.5%: {r336_hgb['ci_rmse_95'][0]:.1f}")
    axes[1, 1].axvline(r336_hgb['ci_rmse_95'][1], color="red", ls=":", lw=1.5, label=f"97.5%: {r336_hgb['ci_rmse_95'][1]:.1f}")
    axes[1, 1].set_title(f"D. HistGBDT vs Persistence [Block 336 (7 ngày)]\nΔRMSE: {r336_hgb['delta_rmse_mean']:+.1f} kW | Thắng: {r336_hgb['rmse_better_pct']:.1f}%", fontweight="bold")
    axes[1, 1].set_xlabel("ΔRMSE (Persistence - HistGBDT) [kW]")
    axes[1, 1].legend()

    fig.suptitle("13. Kiểm định ý nghĩa thống kê Đa chu kỳ (Block 24h & Block 7 ngày) qua Contiguous Moving-Block Bootstrap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "13_moving_block_bootstrap_rmse.png", dpi=200)
    plt.close(fig)

    print("\n" + "=" * 80)
    print("✅ ĐÃ HOÀN THÀNH TOÀN DIỆN CONTIGUOUS MOVING-BLOCK BOOTSTRAP!")
    print(f"📊 Báo cáo JSON: {reports_dir / 'moving_block_bootstrap_summary.json'}")
    print(f"📈 Biểu đồ 2x2: {figures_dir / '13_moving_block_bootstrap_rmse.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
