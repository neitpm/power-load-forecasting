# -*- coding: utf-8 -*-
"""
=============================================================================
HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH BASELINE DỰ BÁO PHỤ TẢI ĐIỆN 30 PHÚT
=============================================================================
Bao gồm:
1. Persistence Naive (Chuẩn đánh giá cơ sở tức thời t)
2. Seasonal Naive 24h (Chu kỳ ngày: t - 48 bước)
3. Seasonal Naive 7 ngày (Chu kỳ tuần: t - 336 bước)
4. Rolling Moving Average (Trung bình trượt 3 bước = 1.5 giờ)
5. Ridge Regression (Mô hình tuyến tính phạt L2)
6. LightGBM Regressor (Mô hình Gradient Boosting hiệu năng cao)
7. XGBoost Regressor (Mô hình cây tăng cường phân tán)
8. LightGBM Classifier (Phân loại xu hướng phụ tải: TĂNG / GIẢM / KHÔNG ĐỔI)
=============================================================================
Đánh giá trên tập TEST hoàn toàn độc lập (Zero Data Leakage).
Chuẩn hóa đơn vị kép: kW và MW.
Xuất bảng tổng hợp so sánh và các biểu đồ đánh giá vào:
- output_preprocessing/reports/baseline_models_comparison.csv
- output_preprocessing/reports/baseline_models_comparison.json
- output_preprocessing/figures/09_baseline_predictions_comparison.png
- output_preprocessing/figures/10_baseline_residuals_distribution.png
- output_preprocessing/figures/11_feature_importance_lightgbm.png
- output_preprocessing/figures/12_confusion_matrix_direction.png
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

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb

# Cấu hình thẩm mỹ đồ thị
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Tính sai số đối xứng sMAPE (%)"""
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Đánh giá toàn diện các độ đo hồi quy theo cả kW và MW"""
    # Lọc bỏ các vị trí NaN trong y_true hoặc y_pred nếu có
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]

    mae_kw = float(mean_absolute_error(yt, yp))
    rmse_kw = float(np.sqrt(mean_squared_error(yt, yp)))
    smape = calculate_smape(yt, yp)
    r2 = float(r2_score(yt, yp))

    mae_mw = mae_kw / 1000.0
    rmse_mw = rmse_kw / 1000.0

    return {
        "model_name": model_name,
        "samples_evaluated": int(len(yt)),
        "MAE_kW": round(mae_kw, 2),
        "RMSE_kW": round(rmse_kw, 2),
        "MAE_MW": round(mae_mw, 4),
        "RMSE_MW": round(rmse_mw, 4),
        "sMAPE_pct": round(smape, 2),
        "R2_score": round(r2, 4)
    }


def load_datasets(data_dir: Path):
    print("\n[1/5] Đang đọc tập dữ liệu Train, Validation, Test từ thư mục data...")
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {p.resolve()}. Hãy chạy pipeline tiền xử lý trước!")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    train_df["TIME"] = pd.to_datetime(train_df["TIME"])
    val_df["TIME"] = pd.to_datetime(val_df["TIME"])
    test_df["TIME"] = pd.to_datetime(test_df["TIME"])

    print(f"      -> Train: {len(train_df):,} mẫu ({train_df['TIME'].min()} -> {train_df['TIME'].max()})")
    print(f"      -> Val:   {len(val_df):,} mẫu ({val_df['TIME'].min()} -> {val_df['TIME'].max()})")
    print(f"      -> Test:  {len(test_df):,} mẫu ({test_df['TIME'].min()} -> {test_df['TIME'].max()})")
    return train_df, val_df, test_df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    candidate_features = [
        # Thông số đo điện 3 pha & công suất
        "UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr",
        "U_avg", "U_max", "U_min", "U_unbalance_pct",
        "I_avg", "I_max", "I_min", "I_unbalance_pct",
        "S_kVA",
        # Lịch sử trễ (Lags kW)
        "lag_1", "lag_2", "lag_3", "lag_48", "lag_336",
        # Thống kê trượt (Rolling)
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_48",
        # Đặc trưng chu kỳ thời gian
        "hour", "dayofweek", "dayofmonth", "month", "is_weekend",
        "sin_hour", "cos_hour"
    ]
    selected = [c for c in candidate_features if c in df.columns]
    return selected


def run_baseline_benchmarks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reports_dir: Path,
    figures_dir: Path
):
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_col = "target_P_kW"
    feature_cols = select_feature_columns(train_df)
    print(f"\n[2/5] Số lượng đặc trưng đầu vào được chọn: {len(feature_cols)} đặc trưng")
    print(f"      Danh sách đặc trưng: {feature_cols}")

    # Chuẩn bị dữ liệu cho mô hình
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    results = []
    predictions = {}
    predictions["y_true"] = y_test.values

    # -------------------------------------------------------------------------
    # Baseline 1: Persistence Naive (y_hat[t+1] = P[t])
    # -------------------------------------------------------------------------
    print("\n[3/5] Đang huấn luyện & đánh giá các mô hình Baseline trên tập Test...")
    print("      -> [Model 1] Persistence Naive (y_hat = P_t)...")
    pred_persistence = test_df["P_kW"].values
    res_pers = evaluate_regression(y_test.values, pred_persistence, "1. Persistence Naive (P_t)")
    results.append(res_pers)
    predictions["Persistence"] = pred_persistence

    # -------------------------------------------------------------------------
    # Baseline 2: Seasonal Naive 24h (y_hat[t+1] = lag_48)
    # -------------------------------------------------------------------------
    print("      -> [Model 2] Seasonal Naive 24h (y_hat = lag_48)...")
    pred_s48 = test_df["lag_48"].values
    res_s48 = evaluate_regression(y_test.values, pred_s48, "2. Seasonal Naive 24h (lag_48)")
    results.append(res_s48)
    predictions["Seasonal_24h"] = pred_s48

    # -------------------------------------------------------------------------
    # Baseline 3: Seasonal Naive 7 ngày (y_hat[t+1] = lag_336)
    # -------------------------------------------------------------------------
    print("      -> [Model 3] Seasonal Naive 7 ngày (y_hat = lag_336)...")
    pred_s336 = test_df["lag_336"].values
    res_s336 = evaluate_regression(y_test.values, pred_s336, "3. Seasonal Naive 7d (lag_336)")
    results.append(res_s336)
    predictions["Seasonal_7d"] = pred_s336

    # Chuẩn bị ma trận float32 liên tục để tránh xung đột ctypes trên Windows với NumPy 2.x
    X_train_f32 = np.ascontiguousarray(X_train.values, dtype=np.float32)
    y_train_f32 = np.ascontiguousarray(y_train.values, dtype=np.float32)
    X_val_f32 = np.ascontiguousarray(X_val.values, dtype=np.float32)
    y_val_f32 = np.ascontiguousarray(y_val.values, dtype=np.float32)
    X_test_f32 = np.ascontiguousarray(X_test.values, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Baseline 4: Moving Average 3 bước (y_hat[t+1] = rolling_mean_3)
    # -------------------------------------------------------------------------
    print("      -> [Model 4] Rolling Moving Average 3 steps (1.5h)...")
    pred_ma3 = test_df["rolling_mean_3"].values
    res_ma3 = evaluate_regression(y_test.values, pred_ma3, "4. Moving Average (1.5h)")
    results.append(res_ma3)
    predictions["Moving_Average"] = pred_ma3

    # -------------------------------------------------------------------------
    # Baseline 5: Ridge Regression
    # -------------------------------------------------------------------------
    print("      -> [Model 5] Ridge Linear Regression (alpha=1.0)...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_imp, y_train)
    pred_ridge = ridge.predict(X_test_imp)
    res_ridge = evaluate_regression(y_test.values, pred_ridge, "5. Ridge Regression")
    results.append(res_ridge)
    predictions["Ridge"] = pred_ridge

    # -------------------------------------------------------------------------
    # Baseline 6: XGBoost Regressor
    # -------------------------------------------------------------------------
    print("      -> [Model 6] XGBoost Regressor (Gradient Boosted Trees)...")
    t0 = time.time()
    xgb_reg = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        verbosity=0
    )
    xgb_reg.fit(
        X_train_f32, y_train_f32,
        eval_set=[(X_val_f32, y_val_f32)],
        verbose=False
    )
    t_xgb = time.time() - t0
    pred_xgb = xgb_reg.predict(X_test_f32)
    res_xgb = evaluate_regression(y_test.values, pred_xgb, "6. XGBoost Regressor")
    res_xgb["training_time_sec"] = round(t_xgb, 2)
    results.append(res_xgb)
    predictions["XGBoost"] = pred_xgb

    # -------------------------------------------------------------------------
    # Baseline 7: LightGBM Regressor (hoặc HistGradientBoosting fallback)
    # -------------------------------------------------------------------------
    print("      -> [Model 7] LightGBM Regressor...")
    t0 = time.time()
    try:
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_reg.fit(
            X_train_f32, y_train_f32,
            eval_set=[(X_val_f32, y_val_f32)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        pred_lgb = lgb_reg.predict(X_test_f32)
        model_feat_imp = lgb_reg.feature_importances_
        best_gbdt_name = "LightGBM"
    except Exception as e:
        print(f"         [Fallback to HistGradientBoosting: {e}]")
        from sklearn.ensemble import HistGradientBoostingRegressor
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
        hgb.fit(X_train_f32, y_train_f32)
        pred_lgb = hgb.predict(X_test_f32)
        model_feat_imp = xgb_reg.feature_importances_
        best_gbdt_name = "HistGradientBoosting"

    t_lgb = time.time() - t0
    res_lgb = evaluate_regression(y_test.values, pred_lgb, f"7. {best_gbdt_name} Regressor")
    res_lgb["training_time_sec"] = round(t_lgb, 2)
    results.append(res_lgb)
    predictions["LightGBM"] = pred_lgb

    # Tổng kết bảng đánh giá hồi quy
    df_results = pd.DataFrame(results)
    df_results.to_csv(reports_dir / "baseline_models_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("BẢNG TỔNG KẾT ĐỐI CHIẾU CÁC MÔ HÌNH BASELINE TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 90)
    print(df_results[["model_name", "MAE_kW", "RMSE_kW", "MAE_MW", "RMSE_MW", "sMAPE_pct", "R2_score"]].to_string(index=False))
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Baseline 8: Phân loại xu hướng (TĂNG / GIẢM / KHÔNG ĐỔI)
    # -------------------------------------------------------------------------
    print("\n[4/5] Đang huấn luyện mô hình phân loại xu hướng phụ tải (Directional Classification)...")
    label_map = {"KHÔNG ĐỔI": 0, "TĂNG": 1, "GIẢM": 2}

    y_train_clf = train_df["target_direction"].map(label_map).values
    y_val_clf = val_df["target_direction"].map(label_map).values
    y_test_clf = test_df["target_direction"].map(label_map).values

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="mlogloss"
    )
    xgb_clf.fit(
        X_train_f32, y_train_clf,
        eval_set=[(X_val_f32, y_val_clf)],
        verbose=False
    )
    pred_clf = xgb_clf.predict(X_test_f32)
    acc = accuracy_score(y_test_clf, pred_clf)
    clf_report_dict = classification_report(y_test_clf, pred_clf, target_names=["KHÔNG ĐỔI", "TĂNG", "GIẢM"], output_dict=True)
    print(f"      -> Độ chính xác phân loại xu hướng (Accuracy): {acc * 100:.2f}%")

    # Lưu báo cáo JSON đầy đủ
    full_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "test_samples": len(test_df),
        "regression_benchmarks": results,
        "classification_benchmark": {
            "model_name": "XGBoost Classifier (Direction)",
            "accuracy_pct": round(acc * 100, 2),
            "report": clf_report_dict
        }
    }
    with open(reports_dir / "baseline_models_comparison.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # 5. SINH BỘ 4 BIỂU ĐỒ BÁO CÁO MÔ HÌNH (DPI 200 - CHUẨN KHOA HỌC)
    # -------------------------------------------------------------------------
    print("\n[5/5] Đang tạo 4 biểu đồ đánh giá mô hình chuẩn khoa học vào figures/...")

    # Biểu đồ 9: So sánh Actual vs Predicted (Zoom 7 ngày cuối cùng)
    fig9, ax9 = plt.subplots(figsize=(16, 6))
    zoom_slice = slice(-336, None)  # 7 ngày (336 bước 30 phút)
    time_zoom = test_df["TIME"].iloc[zoom_slice]

    ax9.plot(time_zoom, y_test.iloc[zoom_slice], label="Thực tế (Ground Truth)", color="black", lw=2.0)
    ax9.plot(time_zoom, predictions["Persistence"][zoom_slice], label="Persistence Naive", color="#7f7f7f", ls="--", lw=1.2, alpha=0.7)
    ax9.plot(time_zoom, predictions["XGBoost"][zoom_slice], label="XGBoost Regressor", color="#ff7f0e", lw=1.6, alpha=0.9)
    ax9.plot(time_zoom, predictions["LightGBM"][zoom_slice], label="LightGBM Regressor", color="#2ca02c", lw=1.6, alpha=0.9)

    ax9.set_title("9. Đối chiếu kết quả dự báo phụ tải P (kW) trên tập Test độc lập (Khung 7 ngày)", fontsize=13, fontweight="bold")
    ax9.set_xlabel("Thời gian")
    ax9.set_ylabel("Công suất P (kW)")
    ax9.legend(loc="upper right", frameon=True)
    fig9.tight_layout()
    fig9.savefig(figures_dir / "09_baseline_predictions_comparison.png", dpi=200)
    plt.close(fig9)

    # Biểu đồ 10: Phân phối phần dư sai số (Residuals Distribution)
    fig10, axes10 = plt.subplots(1, 2, figsize=(15, 5))
    res_lgb_vals = y_test.values - predictions["LightGBM"]
    res_xgb_vals = y_test.values - predictions["XGBoost"]

    sns.histplot(res_lgb_vals, bins=60, kde=True, ax=axes10[0], color="#2ca02c", alpha=0.6)
    axes10[0].axvline(0, color="red", ls="--", lw=1.5)
    axes10[0].set_title(f"Phần dư LightGBM (MAE: {res_lgb['MAE_kW']} kW | {res_lgb['MAE_MW']} MW)", fontsize=12, fontweight="bold")
    axes10[0].set_xlabel("Sai số phần dư = P_true - P_pred (kW)")

    sns.histplot(res_xgb_vals, bins=60, kde=True, ax=axes10[1], color="#ff7f0e", alpha=0.6)
    axes10[1].axvline(0, color="red", ls="--", lw=1.5)
    axes10[1].set_title(f"Phần dư XGBoost (MAE: {res_xgb['MAE_kW']} kW | {res_xgb['MAE_MW']} MW)", fontsize=12, fontweight="bold")
    axes10[1].set_xlabel("Sai số phần dư = P_true - P_pred (kW)")

    fig10.suptitle("10. Kiểm định phân phối phần dư sai số dự báo (Residuals Analysis)", fontsize=14, fontweight="bold")
    fig10.tight_layout()
    fig10.savefig(figures_dir / "10_baseline_residuals_distribution.png", dpi=200)
    plt.close(fig10)

    # Biểu đồ 11: Feature Importance (Top 15 đặc trưng quan trọng nhất)
    fig11, ax11 = plt.subplots(figsize=(10, 6))
    importances = pd.Series(model_feat_imp, index=feature_cols).sort_values(ascending=True)
    top15 = importances.tail(15)
    bars11 = ax11.barh(top15.index, top15.values, color=sns.color_palette("viridis", len(top15)))
    ax11.set_title(f"11. Mức độ quan trọng của đặc trưng ({best_gbdt_name} Feature Importance - Top 15)", fontsize=13, fontweight="bold")
    ax11.set_xlabel("Độ quan trọng (Importance Score)")
    for b in bars11:
        w = b.get_width()
        ax11.text(w + 0.005 if w < 1.0 else w + 5, b.get_y() + b.get_height()/2, f"{w:.3f}" if w < 1.0 else f"{int(w)}", va="center", fontsize=9, fontweight="bold")
    fig11.tight_layout()
    fig11.savefig(figures_dir / "11_feature_importance_gbdt.png", dpi=200)
    plt.close(fig11)

    # Biểu đồ 12: Ma trận nhầm lẫn phân loại xu hướng (Confusion Matrix)
    fig12, ax12 = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test_clf, pred_clf)
    labels = ["KHÔNG ĐỔI", "TĂNG", "GIẢM"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax12)
    ax12.set_title(f"12. Ma trận nhầm lẫn xu hướng phụ tải (Độ chính xác: {acc*100:.2f}%)", fontsize=13, fontweight="bold")
    ax12.set_xlabel("Nhãn dự đoán")
    ax12.set_ylabel("Nhãn thực tế")
    fig12.tight_layout()
    fig12.savefig(figures_dir / "12_confusion_matrix_direction.png", dpi=200)
    plt.close(fig12)

    print("\n" + "=" * 80)
    print("✅ ĐÃ HOÀN THÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ ĐẦY ĐỦ CÁC MÔ HÌNH BASELINE!")
    print(f"📊 Báo cáo CSV: {reports_dir / 'baseline_models_comparison.csv'}")
    print(f"📊 Báo cáo JSON: {reports_dir / 'baseline_models_comparison.json'}")
    print(f"📈 4 Biểu đồ mô hình được lưu tại: {figures_dir}")
    print("=" * 80)
    return df_results


if __name__ == "__main__":
    base_dir = Path("output_preprocessing")
    data_dir = base_dir / "data"
    reports_dir = base_dir / "reports"
    figures_dir = base_dir / "figures"

    train_df, val_df, test_df = load_datasets(data_dir)
    run_baseline_benchmarks(train_df, val_df, test_df, reports_dir, figures_dir)
