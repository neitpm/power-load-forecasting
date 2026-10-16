# -*- coding: utf-8 -*-
"""
=============================================================================
MÔ HÌNH KOLMOGOROV-ARNOLD NETWORK (KAN) CHO DỰ BÁO PHỤ TẢI ĐIỆN 30 PHÚT
BẢN NÂNG CẤP: CHUẨN HÓA MỤC TIÊU (TARGET SCALING) + RESIDUAL LEARNING
=============================================================================
Nguyên lý kỹ thuật:
1. Target Scaling: target_P_kW có giá trị trung bình 8,240 kW. Nếu không chuẩn hóa y,
   mạng nơ-ron sâu với hàm kích hoạt spline/silu biên độ [-1, 1] phải học hệ số khuếch
   đại khổng lồ qua hàng trăm epochs mới tới được dải 8,000.
2. Residual Learning: Mạng KAN học sai số delta: y_pred = P_t + KAN(X)
   (Dự báo giá trị gia số tải thay vì dự báo toàn phần).
3. Đánh giá trên CHÍNH XÁC 14,385 mẫu Test đồng nhất.
4. Kiểm định Contiguous Moving-Block Bootstrap (Block 48 & Block 336).
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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from temporal_significance_test import contiguous_moving_block_bootstrap

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_base=1.0, scale_spline=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = 2.0 / grid_size
        grid = torch.linspace(-1.0 - spline_order * h, 1.0 + spline_order * h, grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features * (grid_size + spline_order)))

        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[: -(k + 1)]) / (grid[k:-1] - grid[: -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[k + 1 :] - x) / (grid[k + 1 :] - grid[1:(-k)]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def forward(self, x):
        base_output = F.linear(F.silu(x), self.base_weight)
        x_clamped = torch.tanh(x)
        spline_basis = self.b_splines(x_clamped).view(x.size(0), -1)
        spline_output = F.linear(spline_basis, self.spline_weight)
        return self.scale_base * base_output + self.scale_spline * spline_output


class ResidualKANRegressor(nn.Module):
    """
    Mạng KAN học phần bù thặng dư (Residual KAN):
    y_hat_scaled = P_t_scaled + KAN(X)
    """
    def __init__(self, in_features, hidden_dims=[64, 32], p_t_idx=0, dropout=0.05):
        super().__init__()
        self.p_t_idx = p_t_idx
        layers = []
        dims = [in_features] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            layers.append(KANLinear(dims[i], dims[i+1], grid_size=5, spline_order=3))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.Dropout(dropout))
        self.kan_body = nn.Sequential(*layers)

    def forward(self, x):
        # x[:, self.p_t_idx] là P_t đã chuẩn hóa
        delta = self.kan_body(x).squeeze(-1)
        base = x[:, self.p_t_idx]
        return base + delta


def evaluate_metrics(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-5)) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {
        "model_name": model_name,
        "MAE_kW": round(float(mae), 2),
        "RMSE_kW": round(float(rmse), 2),
        "MAE_MW": round(float(mae / 1000.0), 4),
        "RMSE_MW": round(float(rmse / 1000.0), 4),
        "sMAPE_pct": round(float(smape), 2),
        "R2_score": round(float(r2), 4)
    }


def main():
    base_dir = Path("output_preprocessing")
    data_dir = base_dir / "data"
    reports_dir = base_dir / "reports"
    figures_dir = base_dir / "figures"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HUẤN LUYỆN RESIDUAL KOLMOGOROV-ARNOLD NETWORK (RESIDUAL KAN)")
    print("=" * 80)

    # 1. Đọc dữ liệu
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    test_df["TIME"] = pd.to_datetime(test_df["TIME"])
    test_df = test_df.sort_values("TIME").reset_index(drop=True)

    candidate_features = [
        "P_kW", # Đặt P_kW ở vị trí đầu tiên (index 0) làm cơ sở residual
        "UA", "UB", "UC", "IA", "IB", "IC", "CosF", "Q_kVAr",
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
    y_test_raw = test_df[target_col].values

    mask_common = (
        np.isfinite(y_test_raw) &
        np.isfinite(pred_persistence) &
        np.isfinite(pred_s48) &
        np.isfinite(pred_s336) &
        np.isfinite(pred_ma3)
    )

    # 2. Xử lý thiếu & chuẩn hóa độc lập (Zero Leakage)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(train_df[feature_cols])
    y_train = train_df[target_col].values

    X_val_imp = imputer.transform(val_df[feature_cols])
    y_val = val_df[target_col].values

    X_test_imp = imputer.transform(test_df[feature_cols])
    y_test = test_df[target_col].values

    # Chuẩn hóa đồng nhất X và Y theo thông số phân phối của P_kW tập Train
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_imp)
    X_val_scaled = scaler_X.transform(X_val_imp)
    X_test_scaled = scaler_X.transform(X_test_imp)

    # Dùng mean và scale của P_kW (cột 0) để chuẩn hóa Y đảm bảo scale delta khớp hoàn toàn
    p_mean = scaler_X.mean_[0]
    p_std = scaler_X.scale_[0]

    y_train_scaled = (y_train - p_mean) / p_std
    y_val_scaled = (y_val - p_mean) / p_std

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_scaled, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # 3. Khởi tạo mô hình Residual KAN
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    model = ResidualKANRegressor(in_features=len(feature_cols), hidden_dims=[64, 32], p_t_idx=0, dropout=0.05).to(device)
    criterion = nn.HuberLoss(delta=0.1) # Chuẩn hóa trong không gian scaled
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    epochs = 12
    best_val_loss = float("inf")
    best_model_state = None

    print(f"[3/5] Bắt đầu huấn luyện Residual KAN ({epochs} epochs)...")
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

        print(f"      Epoch {ep+1:02d}/{epochs} | Scaled Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    train_time = time.time() - t0
    model.load_state_dict(best_model_state)

    # 4. Dự báo trên tập Test độc lập
    print("\n[4/5] Đánh giá trên 14,385 mẫu Test đồng nhất...")
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_pred_scaled = model(test_tensor).numpy()

    # Chuyển ngược về đơn vị thực tế (kW)
    y_pred_all = y_pred_scaled * p_std + p_mean

    y_eval = y_test[mask_common]
    p_eval = pred_persistence[mask_common]
    kan_eval = y_pred_all[mask_common]

    kan_metrics = evaluate_metrics(y_eval, kan_eval, "Residual Kolmogorov-Arnold Network (KAN)")
    kan_metrics["training_time_sec"] = round(train_time, 2)
    kan_metrics["samples_evaluated"] = int(len(y_eval))

    pers_metrics = evaluate_metrics(y_eval, p_eval, "Persistence Naive (P_t)")

    print("\n" + "=" * 90)
    print("BẢNG KẾT QUẢ ĐỐI CHIẾU TRÊN 14,385 MẪU TEST:")
    print("=" * 90)
    df_res = pd.DataFrame([pers_metrics, kan_metrics])
    print(df_res[["model_name", "MAE_kW", "RMSE_kW", "MAE_MW", "RMSE_MW", "sMAPE_pct", "R2_score"]].to_string(index=False))
    print("=" * 90)

    # 5. Kiểm định thống kê Contiguous M-BB (Block 48 & Block 336)
    print("\n[5/5] Kiểm định thống kê Contiguous Moving-Block Bootstrap...")
    eval_df = test_df[mask_common].copy().reset_index(drop=True)
    diffs = eval_df["TIME"].diff()
    segment_ids = (diffs != pd.Timedelta(minutes=30)).cumsum().values

    mbb_48 = contiguous_moving_block_bootstrap(y_eval, kan_eval, p_eval, segment_ids, block_length=48, n_boot=1000)
    mbb_336 = contiguous_moving_block_bootstrap(y_eval, kan_eval, p_eval, segment_ids, block_length=336, n_boot=1000)

    print(f"\nResidual KAN vs Persistence [Block 48 (24h)]:")
    print(f"  ΔRMSE trung bình: {mbb_48['delta_rmse_mean']:+.2f} kW | 95% CI: [{mbb_48['ci_rmse_95'][0]:.2f}; {mbb_48['ci_rmse_95'][1]:.2f}] kW")
    print(f"  Tỷ lệ thắng RMSE: {mbb_48['rmse_better_pct']:.1f}%")
    print(f"  ΔMAE trung bình:  {mbb_48['delta_mae_mean']:+.2f} kW | 95% CI: [{mbb_48['ci_mae_95'][0]:.2f}; {mbb_48['ci_mae_95'][1]:.2f}] kW")

    print(f"\nResidual KAN vs Persistence [Block 336 (7 ngày)]:")
    print(f"  ΔRMSE trung bình: {mbb_336['delta_rmse_mean']:+.2f} kW | 95% CI: [{mbb_336['ci_rmse_95'][0]:.2f}; {mbb_336['ci_rmse_95'][1]:.2f}] kW")
    print(f"  Tỷ lệ thắng RMSE: {mbb_336['rmse_better_pct']:.1f}%")
    print(f"  ΔMAE trung bình:  {mbb_336['delta_mae_mean']:+.2f} kW | 95% CI: [{mbb_336['ci_mae_95'][0]:.2f}; {mbb_336['ci_mae_95'][1]:.2f}] kW")

    # Lưu kết quả
    kan_summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "architecture": "Residual KAN (y_hat = P_t + KAN_delta) with B-Splines (order=3, grid=5)",
        "hidden_dims": [64, 32],
        "input_features": len(feature_cols),
        "evaluated_samples": len(y_eval),
        "metrics": kan_metrics,
        "mbb_significance_test": {
            "block_48_steps_24h": {
                "delta_rmse_mean_kW": round(float(mbb_48['delta_rmse_mean']), 2),
                "ci_rmse_95_kW": [round(float(x), 2) for x in mbb_48['ci_rmse_95']],
                "rmse_better_pct": round(float(mbb_48['rmse_better_pct']), 1),
                "delta_mae_mean_kW": round(float(mbb_48['delta_mae_mean']), 2),
                "ci_mae_95_kW": [round(float(x), 2) for x in mbb_48['ci_mae_95']],
                "is_significant_at_005": bool(mbb_48['ci_rmse_95'][0] > 0)
            },
            "block_336_steps_7d": {
                "delta_rmse_mean_kW": round(float(mbb_336['delta_rmse_mean']), 2),
                "ci_rmse_95_kW": [round(float(x), 2) for x in mbb_336['ci_rmse_95']],
                "rmse_better_pct": round(float(mbb_336['rmse_better_pct']), 1),
                "delta_mae_mean_kW": round(float(mbb_336['delta_mae_mean']), 2),
                "ci_mae_95_kW": [round(float(x), 2) for x in mbb_336['ci_mae_95']],
                "is_significant_at_005": bool(mbb_336['ci_rmse_95'][0] > 0)
            }
        }
    }

    with open(reports_dir / "kan_model_results.json", "w", encoding="utf-8") as f:
        json.dump(kan_summary, f, indent=4, ensure_ascii=False)

    # Vẽ biểu đồ
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    zoom_slice = slice(-336, None)
    time_zoom = test_df["TIME"].iloc[zoom_slice]

    axes[0].plot(time_zoom, y_test[zoom_slice], label="Thực tế (Ground Truth)", color="black", lw=2.0)
    axes[0].plot(time_zoom, pred_persistence[zoom_slice], label="Persistence Naive", color="#7f7f7f", ls="--", lw=1.2, alpha=0.7)
    axes[0].plot(time_zoom, y_pred_all[zoom_slice], label=f"Residual KAN (MAE: {kan_metrics['MAE_kW']} kW | RMSE: {kan_metrics['RMSE_kW']} kW)", color="#d62728", lw=1.6, alpha=0.9)
    axes[0].set_title("14A. Đối chiếu kết quả dự báo phụ tải của Mạng Residual KAN trên tập Test độc lập (Khung 7 ngày)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Thời gian")
    axes[0].set_ylabel("Công suất P (kW)")
    axes[0].legend(loc="upper right", frameon=True)

    residuals_kan = y_eval - kan_eval
    sns.histplot(residuals_kan, bins=60, kde=True, ax=axes[1], color="#d62728", alpha=0.6)
    axes[1].axvline(0, color="black", ls="--", lw=1.5, label="Sai số = 0 kW")
    axes[1].axvline(np.mean(residuals_kan), color="blue", ls=":", lw=1.5, label=f"Sai số trung bình: {np.mean(residuals_kan):.2f} kW")
    axes[1].set_title(f"14B. Phân phối sai số phần dư của Mạng Residual KAN (MAE: {kan_metrics['MAE_kW']} kW | R2: {kan_metrics['R2_score']:.4f})", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Sai số phần dư = P_true - P_pred (kW)")
    axes[1].legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(figures_dir / "14_kan_predictions_and_residuals.png", dpi=200)
    plt.close(fig)

    print("\n✅ Đã lưu kết quả Residual KAN vào:")
    print(f"   - {reports_dir / 'kan_model_results.json'}")
    print(f"   - {figures_dir / '14_kan_predictions_and_residuals.png'}")


if __name__ == "__main__":
    main()
