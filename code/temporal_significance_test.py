# -*- coding: utf-8 -*-
"""
=============================================================================
KIỂM ĐỊNH THỐNG KÊ Ý NGHĨA VỚI MOVING-BLOCK BOOTSTRAP (M-BB)
=============================================================================
Tôn trọng cấu trúc tự tương quan chuỗi thời gian (Autocorrelation):
- Block size = 48 bước (tương đương chu kỳ ngày 24h)
- Số lần lặp Bootstrap B = 1,000 lần
- Tính khoảng tin cậy 95% CI cho ΔRMSE và ΔMAE giữa GBDT và Persistence Naive
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

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def moving_block_bootstrap(y_true, y_pred_cand, y_pred_base, block_length=48, n_boot=1000, seed=42):
    np.random.seed(seed)
    n = len(y_true)
    num_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length + 1

    diff_rmse_list = []
    diff_mae_list = []

    rmse_base_orig = np.sqrt(mean_squared_error(y_true, y_pred_base))
    rmse_cand_orig = np.sqrt(mean_squared_error(y_true, y_pred_cand))
    mae_base_orig = mean_absolute_error(y_true, y_pred_base)
    mae_cand_orig = mean_absolute_error(y_true, y_pred_cand)

    for b in range(n_boot):
        start_indices = np.random.randint(0, max_start, size=num_blocks)
        boot_indices = np.concatenate([np.arange(idx, idx + block_length) for idx in start_indices])[:n]

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
    print("CHẠY KIỂM ĐỊNH MOVING-BLOCK BOOTSTRAP (BLOCK 48 BƯỚC = 24H)")
    print("=" * 80)

    # 1. Đọc test.csv
    test_df = pd.read_csv(data_dir / "test.csv")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")

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

    y_eval = y_test[mask_common]
    p_eval = pred_persistence[mask_common]

    # Huấn luyện nhanh XGBoost để lấy dự báo
    from sklearn.impute import SimpleImputer
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingRegressor

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(train_df[feature_cols])
    y_train = train_df[target_col].values
    X_val_imp = imputer.transform(val_df[feature_cols])
    y_val = val_df[target_col].values
    X_test_imp = imputer.transform(test_df[feature_cols])

    print("Huấn luyện XGBoost...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, early_stopping_rounds=30
    )
    model_xgb.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], verbose=False)
    pred_xgb = model_xgb.predict(X_test_imp)[mask_common]

    print("Huấn luyện HistGradientBoosting...")
    model_hgb = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=6, random_state=42, early_stopping=True
    )
    model_hgb.fit(X_train_imp, y_train)
    pred_hgb = model_hgb.predict(X_test_imp)[mask_common]

    print(f"\nSố mẫu đánh giá chuỗi thời gian liên tục: {len(y_eval):,} mẫu.")

    # Chạy Moving-Block Bootstrap
    print("\n--- Moving-Block Bootstrap (XGBoost vs Persistence) ---")
    res_xgb = moving_block_bootstrap(y_eval, pred_xgb, p_eval, block_length=48, n_boot=1000)
    print(f"ΔRMSE trung bình (Persistence - XGBoost): {res_xgb['delta_rmse_mean']:+.2f} kW")
    print(f"95% CI ΔRMSE: [{res_xgb['ci_rmse_95'][0]:.2f}; {res_xgb['ci_rmse_95'][1]:.2f}] kW")
    print(f"Tỷ lệ XGBoost có RMSE tốt hơn Persistence: {res_xgb['rmse_better_pct']:.1f}%")
    print(f"ΔMAE trung bình (Persistence - XGBoost): {res_xgb['delta_mae_mean']:+.2f} kW")
    print(f"95% CI ΔMAE: [{res_xgb['ci_mae_95'][0]:.2f}; {res_xgb['ci_mae_95'][1]:.2f}] kW")

    print("\n--- Moving-Block Bootstrap (HistGBDT vs Persistence) ---")
    res_hgb = moving_block_bootstrap(y_eval, pred_hgb, p_eval, block_length=48, n_boot=1000)
    print(f"ΔRMSE trung bình (Persistence - HistGBDT): {res_hgb['delta_rmse_mean']:+.2f} kW")
    print(f"95% CI ΔRMSE: [{res_hgb['ci_rmse_95'][0]:.2f}; {res_hgb['ci_rmse_95'][1]:.2f}] kW")
    print(f"Tỷ lệ HistGBDT có RMSE tốt hơn Persistence: {res_hgb['rmse_better_pct']:.1f}%")

    # Lưu kết quả
    mbb_summary = {
        "method": "Moving-Block Bootstrap (Block=48 steps / 24h, B=1000 iterations)",
        "evaluated_samples": len(y_eval),
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

    with open(reports_dir / "moving_block_bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(mbb_summary, f, indent=4, ensure_ascii=False)

    # Vẽ biểu đồ phân phối MBB
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(res_xgb["diff_rmse_samples"], bins=40, kde=True, ax=axes[0], color="#ff7f0e", alpha=0.6)
    axes[0].axvline(0, color="black", ls="--", lw=1.5, label="Không chênh lệch (0 kW)")
    axes[0].axvline(res_xgb['ci_rmse_95'][0], color="red", ls=":", lw=1.5, label=f"2.5%: {res_xgb['ci_rmse_95'][0]:.1f}")
    axes[0].axvline(res_xgb['ci_rmse_95'][1], color="red", ls=":", lw=1.5, label=f"97.5%: {res_xgb['ci_rmse_95'][1]:.1f}")
    axes[0].set_title(f"XGBoost vs Persistence (ΔRMSE: {res_xgb['delta_rmse_mean']:+.1f} kW)", fontweight="bold")
    axes[0].set_xlabel("Chênh lệch RMSE = RMSE(Persistence) - RMSE(XGBoost) [kW]")
    axes[0].legend()

    sns.histplot(res_hgb["diff_rmse_samples"], bins=40, kde=True, ax=axes[1], color="#2ca02c", alpha=0.6)
    axes[1].axvline(0, color="black", ls="--", lw=1.5, label="Không chênh lệch (0 kW)")
    axes[1].axvline(res_hgb['ci_rmse_95'][0], color="red", ls=":", lw=1.5, label=f"2.5%: {res_hgb['ci_rmse_95'][0]:.1f}")
    axes[1].axvline(res_hgb['ci_rmse_95'][1], color="red", ls=":", lw=1.5, label=f"97.5%: {res_hgb['ci_rmse_95'][1]:.1f}")
    axes[1].set_title(f"HistGBDT vs Persistence (ΔRMSE: {res_hgb['delta_rmse_mean']:+.1f} kW)", fontweight="bold")
    axes[1].set_xlabel("Chênh lệch RMSE = RMSE(Persistence) - RMSE(HistGBDT) [kW]")
    axes[1].legend()

    fig.suptitle("13. Phân phối chênh lệch RMSE qua 1,000 lần Moving-Block Bootstrap (Block 48 bước)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "13_moving_block_bootstrap_rmse.png", dpi=200)
    plt.close(fig)

    print("\n✅ Đã hoàn thành Moving-Block Bootstrap và lưu kết quả vào:")
    print(f"   - {reports_dir / 'moving_block_bootstrap_summary.json'}")
    print(f"   - {figures_dir / '13_moving_block_bootstrap_rmse.png'}")


if __name__ == "__main__":
    main()
