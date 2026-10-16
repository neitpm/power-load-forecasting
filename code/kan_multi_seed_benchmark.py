# -*- coding: utf-8 -*-
"""
=============================================================================
ĐÁNH GIÁ ĐỘ ỔN ĐỊNH VÀ TÍNH LẶP LẠI CỦA MÔ HÌNH RESIDUAL KAN (MULTI-SEED BENCHMARK)
=============================================================================
Mục tiêu nghiên cứu khoa học:
1. Huấn luyện Residual KAN qua 5 random seeds độc lập: [42, 101, 2024, 7, 999].
2. Đánh giá hoàn toàn khách quan trên đúng 14,385 mẫu Test đồng nhất.
3. Không tinh chỉnh siêu tham số trên Test, cố định hoàn toàn quy trình huấn luyện.
4. Báo cáo:
   - Trung bình (Mean) và Độ lệch chuẩn (Std Dev) của MAE, RMSE, sMAPE, R2.
   - Thống kê chênh lệch ΔRMSE, ΔMAE so với Persistence Naive qua từng seed.
   - Khoảng tin cậy tổng hợp qua Contiguous Moving-Block Bootstrap.
5. Xuất báo cáo khoa học vào:
   - output_preprocessing/reports/kan_multi_seed_stability.csv
   - output_preprocessing/reports/kan_multi_seed_stability.json
   - output_preprocessing/figures/15_kan_multi_seed_stability.png
=============================================================================
"""

import os
import sys
import json
import time
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

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from temporal_significance_test import contiguous_moving_block_bootstrap
from models_kan import ResidualKANRegressor, evaluate_metrics

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def train_single_seed(seed, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, p_mean, p_std, in_features, epochs=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_scaled, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model = ResidualKANRegressor(in_features=in_features, hidden_dims=[64, 32], p_t_idx=0, dropout=0.05).to(device)
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_model_state = None

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                loss = criterion(preds, by)
                val_loss += loss.item() * len(by)
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    train_time = time.time() - t0
    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_pred_scaled = model(test_tensor).numpy()

    y_pred_kw = y_pred_scaled * p_std + p_mean
    return y_pred_kw, train_time


def main():
    base_dir = Path("output_preprocessing")
    data_dir = base_dir / "data"
    reports_dir = base_dir / "reports"
    figures_dir = base_dir / "figures"

    print("=" * 85)
    print("KHẢO SÁT TÍNH ỔN ĐỊNH VÀ LẶP LẠI CỦA RESIDUAL KAN QUA 5 RANDOM SEEDS")
    print("=" * 85)

    # 1. Nạp dữ liệu
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    test_df["TIME"] = pd.to_datetime(test_df["TIME"])
    test_df = test_df.sort_values("TIME").reset_index(drop=True)

    candidate_features = [
        "P_kW", "UA", "UB", "UC", "IA", "IB", "IC", "CosF", "Q_kVAr",
        "U_avg", "U_max", "U_min", "U_unbalance_pct",
        "I_avg", "I_max", "I_min", "I_unbalance_pct",
        "S_kVA", "lag_1", "lag_2", "lag_3", "lag_48", "lag_336",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_48",
        "hour", "dayofweek", "dayofmonth", "month", "is_weekend",
        "sin_hour", "cos_hour"
    ]
    feature_cols = [c for c in candidate_features if c in train_df.columns]
    target_col = "target_P_kW"

    # Lọc mask đồng nhất 14,385 mẫu
    pred_persistence = test_df["P_kW"].values
    pred_s48 = test_df["lag_48"].values
    pred_s336 = test_df["lag_336"].values
    pred_ma3 = test_df["rolling_mean_3"].values
    y_test_raw = test_df[target_col].values

    mask_common = (
        np.isfinite(y_test_raw) &
        np.isfinite(pred_persistence) &
        np.isfinite(pred_s48) &
        np.isfinite(pred_s336) &
        np.isfinite(pred_ma3)
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(train_df[feature_cols])
    y_train = train_df[target_col].values
    X_val_imp = imputer.transform(val_df[feature_cols])
    y_val = val_df[target_col].values
    X_test_imp = imputer.transform(test_df[feature_cols])
    y_test = test_df[target_col].values

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_imp)
    X_val_scaled = scaler_X.transform(X_val_imp)
    X_test_scaled = scaler_X.transform(X_test_imp)

    p_mean = scaler_X.mean_[0]
    p_std = scaler_X.scale_[0]

    y_train_scaled = (y_train - p_mean) / p_std
    y_val_scaled = (y_val - p_mean) / p_std

    y_eval = y_test[mask_common]
    p_eval = pred_persistence[mask_common]

    eval_df = test_df[mask_common].copy().reset_index(drop=True)
    diffs = eval_df["TIME"].diff()
    segment_ids = (diffs != pd.Timedelta(minutes=30)).cumsum().values

    # Chuẩn bị 5 seeds độc lập
    seeds = [42, 101, 2024, 7, 999]
    seed_records = []
    predictions_all_seeds = []

    print(f"Bắt đầu huấn luyện và đánh giá trên {len(seeds)} random seeds: {seeds}...")
    for idx, s in enumerate(seeds, start=1):
        print(f"\n--- [Seed {idx}/{len(seeds)}: random_seed = {s}] ---")
        pred_kw, dur = train_single_seed(
            s, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled,
            p_mean, p_std, in_features=len(feature_cols), epochs=12
        )
        pred_eval = pred_kw[mask_common]
        predictions_all_seeds.append(pred_eval)

        # Tính toán metrics
        metrics = evaluate_metrics(y_eval, pred_eval, f"Residual KAN (Seed {s})")
        metrics["seed"] = s
        metrics["training_time_sec"] = round(dur, 2)

        # Đánh giá cải thiện so với Persistence Naive
        d_mae = pers_metrics_mae = 598.36 - metrics["MAE_kW"]
        d_rmse = pers_metrics_rmse = 1160.44 - metrics["RMSE_kW"]
        metrics["Delta_MAE_kW"] = round(d_mae, 2)
        metrics["Delta_RMSE_kW"] = round(d_rmse, 2)
        metrics["Improvement_MAE_pct"] = round((d_mae / 598.36) * 100.0, 2)
        metrics["Improvement_RMSE_pct"] = round((d_rmse / 1160.44) * 100.0, 2)

        print(f"      MAE: {metrics['MAE_kW']} kW (Tốt hơn Persistence: {metrics['Improvement_MAE_pct']}%)")
        print(f"      RMSE: {metrics['RMSE_kW']} kW (Tốt hơn Persistence: {metrics['Improvement_RMSE_pct']}%)")
        print(f"      R2: {metrics['R2_score']:.4f} | Thời gian: {metrics['training_time_sec']}s")

        seed_records.append(metrics)

    df_seeds = pd.DataFrame(seed_records)

    # Thống kê tổng hợp Mean ± Std
    mean_mae = df_seeds["MAE_kW"].mean()
    std_mae = df_seeds["MAE_kW"].std()

    mean_rmse = df_seeds["RMSE_kW"].mean()
    std_rmse = df_seeds["RMSE_kW"].std()

    mean_r2 = df_seeds["R2_score"].mean()
    std_r2 = df_seeds["R2_score"].std()

    mean_smape = df_seeds["sMAPE_pct"].mean()
    std_smape = df_seeds["sMAPE_pct"].std()

    mean_imp_mae = df_seeds["Improvement_MAE_pct"].mean()
    std_imp_mae = df_seeds["Improvement_MAE_pct"].std()

    mean_imp_rmse = df_seeds["Improvement_RMSE_pct"].mean()
    std_imp_rmse = df_seeds["Improvement_RMSE_pct"].std()

    print("\n" + "=" * 90)
    print("BẢNG TỔNG HỢP ĐỘ ỔN ĐỊNH QUA 5 RANDOM SEEDS ĐỘC LẬP")
    print("=" * 90)
    cols_show = ["seed", "MAE_kW", "RMSE_kW", "sMAPE_pct", "R2_score", "Improvement_MAE_pct", "Improvement_RMSE_pct"]
    print(df_seeds[cols_show].to_string(index=False))
    print("-" * 90)
    print(f"TRUNG BÌNH ± ĐỘ LỆCH CHUẨN (MEAN ± STD):")
    print(f"  * MAE:  {mean_mae:.2f} ± {std_mae:.2f} kW  (Cải thiện: {mean_imp_mae:.2f}% ± {std_imp_mae:.2f}%)")
    print(f"  * RMSE: {mean_rmse:.2f} ± {std_rmse:.2f} kW (Cải thiện: {mean_imp_rmse:.2f}% ± {std_imp_rmse:.2f}%)")
    print(f"  * R2:   {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"  * sMAPE:{mean_smape:.2f}% ± {std_smape:.2f}%")
    print("=" * 90)

    # Lưu bảng CSV
    df_seeds.to_csv(reports_dir / "kan_multi_seed_stability.csv", index=False, encoding="utf-8-sig")

    # Tính Ensemble trung bình của 5 seeds
    ensemble_preds = np.mean(predictions_all_seeds, axis=0)
    ens_metrics = evaluate_metrics(y_eval, ensemble_preds, "Residual KAN (5-Seed Ensemble Mean)")
    ens_metrics["Improvement_MAE_pct"] = round(((598.36 - ens_metrics["MAE_kW"]) / 598.36) * 100.0, 2)
    ens_metrics["Improvement_RMSE_pct"] = round(((1160.44 - ens_metrics["RMSE_kW"]) / 1160.44) * 100.0, 2)

    # Kiểm định thống kê Contiguous MBB trên mô hình Ensemble
    mbb_48_ens = contiguous_moving_block_bootstrap(y_eval, ensemble_preds, p_eval, segment_ids, block_length=48, n_boot=1000)
    mbb_336_ens = contiguous_moving_block_bootstrap(y_eval, ensemble_preds, p_eval, segment_ids, block_length=336, n_boot=1000)

    summary_json = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "seeds_evaluated": seeds,
        "evaluated_samples": int(len(y_eval)),
        "summary_statistics": {
            "MAE_kW_mean": round(float(mean_mae), 2),
            "MAE_kW_std": round(float(std_mae), 2),
            "RMSE_kW_mean": round(float(mean_rmse), 2),
            "RMSE_kW_std": round(float(std_rmse), 2),
            "sMAPE_pct_mean": round(float(mean_smape), 2),
            "sMAPE_pct_std": round(float(std_smape), 2),
            "R2_score_mean": round(float(mean_r2), 4),
            "R2_score_std": round(float(std_r2), 4),
            "Improvement_MAE_pct_mean": round(float(mean_imp_mae), 2),
            "Improvement_RMSE_pct_mean": round(float(mean_imp_rmse), 2)
        },
        "individual_runs": seed_records,
        "ensemble_5_seeds": {
            "metrics": ens_metrics,
            "mbb_significance_block_48": {
                "delta_rmse_mean_kW": round(float(mbb_48_ens['delta_rmse_mean']), 2),
                "ci_rmse_95_kW": [round(float(x), 2) for x in mbb_48_ens['ci_rmse_95']],
                "rmse_better_pct": round(float(mbb_48_ens['rmse_better_pct']), 1),
                "delta_mae_mean_kW": round(float(mbb_48_ens['delta_mae_mean']), 2),
                "ci_mae_95_kW": [round(float(x), 2) for x in mbb_48_ens['ci_mae_95']]
            },
            "mbb_significance_block_336": {
                "delta_rmse_mean_kW": round(float(mbb_336_ens['delta_rmse_mean']), 2),
                "ci_rmse_95_kW": [round(float(x), 2) for x in mbb_336_ens['ci_rmse_95']],
                "rmse_better_pct": round(float(mbb_336_ens['rmse_better_pct']), 1),
                "delta_mae_mean_kW": round(float(mbb_336_ens['delta_mae_mean']), 2),
                "ci_mae_95_kW": [round(float(x), 2) for x in mbb_336_ens['ci_mae_95']]
            }
        }
    }

    with open(reports_dir / "kan_multi_seed_stability.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)

    # 4. Vẽ biểu đồ thẩm mỹ cao thể hiện độ ổn định qua các seeds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Đồ thị MAE qua các Seed
    x_pos = np.arange(len(seeds))
    bars0 = axes[0].bar(x_pos, df_seeds["MAE_kW"], color="#d62728", alpha=0.7, width=0.55, label="Residual KAN (Seed)")
    axes[0].axhline(598.36, color="black", ls="--", lw=1.8, label="Persistence Naive (598.36 kW)")
    axes[0].axhline(mean_mae, color="blue", ls=":", lw=1.8, label=f"KAN Mean ({mean_mae:.1f} ± {std_mae:.1f} kW)")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"Seed {s}" for s in seeds], fontweight="bold")
    axes[0].set_ylabel("MAE (kW)", fontweight="bold")
    axes[0].set_title(f"A. Độ ổn định sai số MAE qua 5 Seeds (Mean: {mean_mae:.1f} kW)", fontweight="bold")
    axes[0].set_ylim(400, 650)
    for b in bars0:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2, h + 5, f"{h:.1f}", ha="center", fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper right")

    # Đồ thị RMSE qua các Seed
    bars1 = axes[1].bar(x_pos, df_seeds["RMSE_kW"], color="#2ca02c", alpha=0.7, width=0.55, label="Residual KAN (Seed)")
    axes[1].axhline(1160.44, color="black", ls="--", lw=1.8, label="Persistence Naive (1,160.44 kW)")
    axes[1].axhline(mean_rmse, color="blue", ls=":", lw=1.8, label=f"KAN Mean ({mean_rmse:.1f} ± {std_rmse:.1f} kW)")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"Seed {s}" for s in seeds], fontweight="bold")
    axes[1].set_ylabel("RMSE (kW)", fontweight="bold")
    axes[1].set_title(f"B. Độ ổn định sai số RMSE qua 5 Seeds (Mean: {mean_rmse:.1f} kW)", fontweight="bold")
    axes[1].set_ylim(800, 1250)
    for b in bars1:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2, h + 10, f"{h:.1f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].legend(loc="upper right")

    fig.suptitle("15. Đánh giá độ ổn định và tính lặp lại của Residual KAN qua 5 Random Seeds trên 14,385 mẫu Test", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "15_kan_multi_seed_stability.png", dpi=200)
    plt.close(fig)

    print("\n✅ ĐÃ HOÀN THÀNH KHẢO SÁT ĐỘ ỔN ĐỊNH VÀ XUẤT CÁC BÁO CÁO:")
    print(f"   - {reports_dir / 'kan_multi_seed_stability.csv'}")
    print(f"   - {reports_dir / 'kan_multi_seed_stability.json'}")
    print(f"   - {figures_dir / '15_kan_multi_seed_stability.png'}")


if __name__ == "__main__":
    main()
