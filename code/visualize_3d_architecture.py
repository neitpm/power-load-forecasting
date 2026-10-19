# -*- coding: utf-8 -*-
"""
=============================================================================
VISUALIZE 3D-STYLE ISOMETRIC ARCHITECTURE FIGURES (300 DPI)
For Scientific ML & Industrial Time-Series Forecasting Portfolio
=============================================================================
Layout Principles:
1. Clear Canvas: 16:9 widescreen (figsize=(14, 8)), generous vertical & horizontal spacing.
2. 2D Isometric projection with explicit depth vector (dx=0.5, dy=0.3) for clean rendering.
3. Strict separation of hierarchy:
   - Header: Category tag, Title, Subtitle at top.
   - Central: Isometric 3D tensor blocks with ample horizontal clearance.
   - Inside/Above Block: Stage / Module Title (Bold, concise).
   - Below Block: Separate dedicated Annotation Box for specifications, shapes, metrics.
4. Zero Overlap: Arrows flow cleanly between blocks without crossing text or edges.
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BG_COLOR = '#0b0f19'
CARD_BG = '#111827'
BORDER_COLOR = '#1f2937'


def draw_isometric_block(ax, x, y, w, h, depth=0.8, color='#2563eb', top_color=None, side_color=None, alpha=0.9, edge_color='#ffffff'):
    """
    Draws a clean 3D isometric cuboid block using 2D polygons to avoid 3D camera overlap.
    x, y: bottom-left of front face
    w: width, h: height
    depth: isometric extrusion amount
    """
    if top_color is None:
        top_color = color
    if side_color is None:
        side_color = color

    # Isometric offsets: dx > 0, dy > 0 (angles up-right)
    iso_dx = depth * 0.7
    iso_dy = depth * 0.45

    # 1. Front face
    front_pts = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])
    front = Polygon(front_pts, closed=True, facecolor=color, edgecolor=edge_color, linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(front)

    # 2. Top face
    top_pts = np.array([
        [x, y + h],
        [x + w, y + h],
        [x + w + iso_dx, y + h + iso_dy],
        [x + iso_dx, y + h + iso_dy]
    ])
    top = Polygon(top_pts, closed=True, facecolor=top_color, edgecolor=edge_color, linewidth=1.5, alpha=min(alpha + 0.1, 1.0), zorder=4)
    ax.add_patch(top)

    # 3. Right side face
    side_pts = np.array([
        [x + w, y],
        [x + w + iso_dx, y + iso_dy],
        [x + w + iso_dx, y + h + iso_dy],
        [x + w, y + h]
    ])
    side = Polygon(side_pts, closed=True, facecolor=side_color, edgecolor=edge_color, linewidth=1.5, alpha=max(alpha - 0.2, 0.4), zorder=3)
    ax.add_patch(side)

    return (x + w/2, y + h/2) # center of front face


def add_header(ax, tag, title, subtitle):
    """Draws standardized high-contrast portfolio header."""
    ax.text(0.5, 0.96, tag.upper(), transform=ax.transAxes, color='#38bdf8', fontsize=10,
            fontweight='bold', ha='center', va='top')
    ax.text(0.5, 0.91, title, transform=ax.transAxes, color='#ffffff', fontsize=16,
            fontweight='bold', ha='center', va='top')
    ax.text(0.5, 0.86, subtitle, transform=ax.transAxes, color='#94a3b8', fontsize=11,
            ha='center', va='top')


# -----------------------------------------------------------------------------
# FIGURE 1: 1_data_pipeline_3d.png
# -----------------------------------------------------------------------------
def make_fig1_data_pipeline():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    add_header(ax, "Data Ingestion & Telemetry Processing Flow",
               "Industrial 3-Phase Telemetry Pipeline & Continuous Segment Extraction",
               "Feeder 480 - Substation 224 Tan Hung (Tay Ninh Power Company) | 2020 - 2025")

    # 4 distinct pipeline stages placed horizontally with wide gaps
    stages = [
        {
            "num": "01",
            "title": "Raw Telemetry\nIngestion",
            "x": 0.8, "y": 3.2, "w": 2.2, "h": 2.2,
            "col": '#0284c7', "top": '#38bdf8', "side": '#0369a1',
            "specs": ["3-Phase U, I, cosφ, P, Q", "30-min sampling interval", "Multi-file quarterly CSVs"]
        },
        {
            "num": "02",
            "title": "Timestamp Grid\nRegularization",
            "x": 4.1, "y": 3.2, "w": 2.2, "h": 2.2,
            "col": '#0d9488', "top": '#2dd4bf', "side": '#0f766e',
            "specs": ["Strict 30-min time index", "Detect missing step intervals", "Wide/Long tensor alignment"]
        },
        {
            "num": "03",
            "title": "Outlier Filtering &\nGap Partitioning",
            "x": 7.4, "y": 3.2, "w": 2.2, "h": 2.2,
            "col": '#d97706', "top": '#fbbf24', "side": '#b45309',
            "specs": ["Physical bounds sanity check", "Telemetry blackout isolator", "Segment break boundaries"]
        },
        {
            "num": "04",
            "title": "Continuous Segments\nCache Construction",
            "x": 10.7, "y": 3.2, "w": 2.2, "h": 2.2,
            "col": '#16a34a', "top": '#4ade80', "side": '#15803d',
            "specs": ["16 continuous test segments", "14,385 valid test records", "No cross-gap lag lookback"]
        }
    ]

    for s in stages:
        # 3D block
        cx, cy = draw_isometric_block(ax, s["x"], s["y"], s["w"], s["h"], depth=0.7,
                                     color=s["col"], top_color=s["top"], side_color=s["side"],
                                     alpha=0.85, edge_color='#ffffff')

        # Text inside front face (Title)
        ax.text(cx, cy + 0.35, s["num"], color='#ffffff', fontsize=16, fontweight='bold', ha='center', va='center', alpha=0.9)
        ax.text(cx, cy - 0.35, s["title"], color='#ffffff', fontsize=10.5, fontweight='bold', ha='center', va='center')

        # Annotation box below block (Specs)
        box_y = 1.3
        bbox_props = dict(boxstyle="round,pad=0.5", fc='#1e293b', ec=s["col"], lw=1.5)
        spec_text = "\n".join([f"• {item}" for item in s["specs"]])
        ax.text(s["x"] + s["w"]/2, box_y, spec_text, color='#e2e8f0', fontsize=9.5, ha='center', va='top', bbox=bbox_props)

    # Clean connecting arrows between blocks
    for x_start in [3.4, 6.7, 10.0]:
        ax.annotate("", xy=(x_start + 0.6, 4.3), xytext=(x_start, 4.3),
                    arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=2.5, mutation_scale=18))

    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_data_pipeline_3d.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print("Saved 1_data_pipeline_3d.png (Cleaned)")


# -----------------------------------------------------------------------------
# FIGURE 2: 2_preprocessing_contract_3d.png
# -----------------------------------------------------------------------------
def make_fig2_preprocessing_contract():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    add_header(ax, "Zero Data Leakage & Temporal Segregation Protocol",
               "Chronological Train / Validation / Test Splitting Contract",
               "Parameters fitted strictly on historical train records | Zero lookahead bias")

    splits = [
        {
            "tag": "TRAINING SPLIT",
            "period": "03/2020 — 01/2024 (46 Months)",
            "x": 0.8, "y": 3.2, "w": 3.2, "h": 2.2,
            "col": '#1d4ed8', "top": '#60a5fa', "side": '#1e40af',
            "contract": ["Fit SimpleImputer (median)", "Fit StandardScaler (μ, σ)", "Model parameters learning"]
        },
        {
            "tag": "VALIDATION SPLIT",
            "period": "01/2024 — 11/2024 (10 Months)",
            "x": 5.2, "y": 3.2, "w": 2.8, "h": 2.2,
            "col": '#b45309', "top": '#fde047', "side": '#92400e',
            "contract": ["Transform only (stored μ, σ)", "Early stopping threshold", "Architecture hyperparameter tuning"]
        },
        {
            "tag": "OUT-OF-SAMPLE TEST",
            "period": "11/2024 — 08/2025 (10 Months)",
            "x": 9.2, "y": 3.2, "w": 3.4, "h": 2.2,
            "col": '#047857', "top": '#34d399', "side": '#065f46',
            "contract": ["14,385 valid evaluation records", "Transform only (zero leakage)", "16 continuous segments for bootstrap"]
        }
    ]

    for sp in splits:
        cx, cy = draw_isometric_block(ax, sp["x"], sp["y"], sp["w"], sp["h"], depth=0.7,
                                     color=sp["col"], top_color=sp["top"], side_color=sp["side"],
                                     alpha=0.85, edge_color='#ffffff')

        # Inside block text
        ax.text(cx, cy + 0.35, sp["tag"], color='#ffffff', fontsize=12, fontweight='bold', ha='center', va='center')
        ax.text(cx, cy - 0.35, sp["period"], color='#e0f2fe', fontsize=9.5, ha='center', va='center')

        # Contract spec below
        box_y = 1.3
        bbox_props = dict(boxstyle="round,pad=0.5", fc='#1e293b', ec=sp["col"], lw=1.5)
        spec_text = "\n".join([f"✓ {c}" for c in sp["contract"]])
        ax.text(sp["x"] + sp["w"]/2, box_y, spec_text, color='#e2e8f0', fontsize=9.5, ha='center', va='top', bbox=bbox_props)

    # Isolation barriers between splits
    for x_bar, label in [(4.4, "BARRIER 1\nNo Future Leakage"), (8.4, "BARRIER 2\nZero Test Optimization")]:
        # Draw red dashed wall
        ax.plot([x_bar, x_bar], [2.2, 5.8], color='#ef4444', lw=2.2, ls='--', zorder=5)
        ax.text(x_bar, 6.0, label, color='#fca5a5', fontsize=8.5, fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle="square,pad=0.25", fc='#450a0a', ec='#ef4444', lw=1.2))

    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_preprocessing_contract_3d.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print("Saved 2_preprocessing_contract_3d.png (Cleaned)")


# -----------------------------------------------------------------------------
# FIGURE 3: 3_model_benchmark_3d.png
# -----------------------------------------------------------------------------
def make_fig3_model_benchmark():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    add_header(ax, "Comparative Out-of-Sample Performance",
               "Error Metric Comparison Across Benchmarked Families (30-min Horizon)",
               "Evaluated on 14,385 out-of-sample test samples | Feeder 480 Tan Hung")

    models = [
        {"name": "Persistence\n(Naive)", "mae": 598.36, "rmse": 1160.44, "col": '#64748b', "top": '#94a3b8', "side": '#475569', "delta": "Baseline (0.00%)"},
        {"name": "Moving\nAverage (1.5h)", "mae": 1016.16, "rmse": 1586.70, "col": '#475569', "top": '#64748b', "side": '#334155', "delta": "-69.82%"},
        {"name": "Ridge\nRegression", "mae": 730.96, "rmse": 1186.89, "col": '#0284c7', "top": '#38bdf8', "side": '#0369a1', "delta": "-22.16%"},
        {"name": "XGBoost\nRegressor", "mae": 645.24, "rmse": 1155.21, "col": '#2563eb', "top": '#60a5fa', "side": '#1d4ed8', "delta": "-7.83%"},
        {"name": "Residual KAN\n(5-Seed Ensemble)", "mae": 519.78, "rmse": 1019.95, "col": '#059669', "top": '#34d399', "side": '#047857', "delta": "+13.13% (Best)"}
    ]

    x_start = 0.8
    step = 2.5

    for i, m in enumerate(models):
        curr_x = x_start + i * step
        # Height scaled proportionally
        h_mae = (m["mae"] / 600.0) * 1.5
        h_rmse = (m["rmse"] / 1200.0) * 2.2

        # 1. RMSE block (taller, behind)
        draw_isometric_block(ax, curr_x + 0.35, 2.5, 0.9, h_rmse, depth=0.5,
                             color=m["col"], top_color=m["top"], side_color=m["side"], alpha=0.4, edge_color='#94a3b8')

        # 2. MAE block (front)
        draw_isometric_block(ax, curr_x, 2.5, 0.9, h_mae, depth=0.5,
                             color=m["col"], top_color=m["top"], side_color=m["side"], alpha=0.9, edge_color='#ffffff')

        # Model label above blocks
        ax.text(curr_x + 0.6, 5.7, m["name"], color='#ffffff', fontsize=10.5, fontweight='bold', ha='center', va='bottom')

        # Score box below blocks
        bbox_props = dict(boxstyle="round,pad=0.4", fc='#1e293b', ec=m["col"], lw=1.5)
        score_text = f"MAE:  {m['mae']:.1f} kW\nRMSE: {m['rmse']:.1f} kW\nΔMAE: {m['delta']}"
        ax.text(curr_x + 0.6, 1.8, score_text, color='#e2e8f0', fontsize=9, ha='center', va='top', bbox=bbox_props)

    # Legend at bottom center
    legend_text = "■ Solid Front: MAE (kW)   |   ▨ Shaded Back: RMSE (kW)   |   Evaluation Set: 14,385 Out-of-sample Records"
    ax.text(0.5, 0.05, legend_text, transform=ax.transAxes, color='#38bdf8', fontsize=10, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc='#111827', ec='#1e293b', lw=1.2))

    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_model_benchmark_3d.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print("Saved 3_model_benchmark_3d.png (Cleaned)")


# -----------------------------------------------------------------------------
# FIGURE 4: 4_kan_ensemble_3d.png
# -----------------------------------------------------------------------------
def make_fig4_kan_ensemble():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    add_header(ax, "Ensemble Architecture & Multi-Seed Stability",
               "Uniform 5-Seed Residual KAN Ensembling Pipeline",
               "Seeds: [42, 101, 2024, 7, 999] | Unweighted Average (Zero Test Overfitting)")

    seeds = [42, 101, 2024, 7, 999]
    y_coords = [5.6, 4.6, 3.6, 2.6, 1.6]

    # Draw 5 individual seed model blocks (Left column)
    for s, y_pos in zip(seeds, y_coords):
        draw_isometric_block(ax, x=1.0, y=y_pos, w=2.6, h=0.7, depth=0.4,
                             color='#4f46e5', top_color='#818cf8', side_color='#3730a3',
                             alpha=0.85, edge_color='#ffffff')
        ax.text(2.3, y_pos + 0.35, f"Residual KAN (Seed {s})", color='#ffffff', fontsize=9.5, fontweight='bold', ha='center', va='center')

        # Arrow from seed to aggregator
        ax.annotate("", xy=(5.2, 3.8), xytext=(4.0, y_pos + 0.35),
                    arrowprops=dict(arrowstyle="->", color='#818cf8', lw=1.8, mutation_scale=14))

    # Central Uniform Aggregator Block
    cx, cy = draw_isometric_block(ax, x=5.3, y=2.5, w=3.4, h=2.5, depth=0.8,
                                 color='#059669', top_color='#34d399', side_color='#047857',
                                 alpha=0.9, edge_color='#ffffff')
    ax.text(cx, cy + 0.45, "UNIFORM ENSEMBLE", color='#ffffff', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(cx, cy - 0.2, "ŷ_ens = (1/5) * Σ ŷ_i\nFixed w_i = 0.2", color='#d1fae5', fontsize=10, ha='center', va='center')

    # Arrow from aggregator to output
    ax.annotate("", xy=(10.0, 3.8), xytext=(9.3, 3.8),
                arrowprops=dict(arrowstyle="->", color='#34d399', lw=2.5, mutation_scale=18))

    # Output Block (Right column)
    ox, oy = draw_isometric_block(ax, x=10.1, y=2.8, w=2.8, h=2.0, depth=0.6,
                                  color='#0284c7', top_color='#38bdf8', side_color='#0369a1',
                                  alpha=0.85, edge_color='#ffffff')
    ax.text(ox, oy + 0.3, "FINAL PREDICTION", color='#ffffff', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(ox, oy - 0.3, "P_pred (t+1)\n[kW]", color='#e0f2fe', fontsize=10, ha='center', va='center')

    # Metric annotation box below
    summary_text = (
        "• Single KAN Stability: MAE = 528.99 ± 12.23 kW | RMSE = 1033.35 ± 13.78 kW (5 Seeds)\n"
        "• Ensemble Performance: MAE = 519.78 kW (-13.13%) | RMSE = 1019.95 kW (-12.11%) | R² = 0.9050\n"
        "• Zero-Leakage Guarantee: No weights tuned on the out-of-sample test split"
    )
    ax.text(0.5, 0.06, summary_text, transform=ax.transAxes, color='#f8fafc', fontsize=9.5, ha='center',
            bbox=dict(boxstyle="round,pad=0.5", fc='#1e293b', ec='#059669', lw=1.5))

    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_kan_ensemble_3d.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print("Saved 4_kan_ensemble_3d.png (Cleaned)")


# -----------------------------------------------------------------------------
# FIGURE 5: 5_evaluation_3d.png
# -----------------------------------------------------------------------------
def make_fig5_evaluation():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    add_header(ax, "Statistical Verification & Multi-Metric Trade-Offs",
               "Evaluation Metrics and Baseline Trade-offs Analysis",
               "Contiguous Moving-Block Bootstrap (B=1,000) & Metric Asymmetry Verification")

    # Block 1: Bootstrap Rigor
    b1_x = 1.2
    draw_isometric_block(ax, x=b1_x, y=2.8, w=4.5, h=2.5, depth=0.7,
                         color='#0284c7', top_color='#38bdf8', side_color='#0369a1',
                         alpha=0.85, edge_color='#ffffff')
    ax.text(b1_x + 2.25, 4.3, "CONTIGUOUS BOOTSTRAP", color='#ffffff', fontsize=12, fontweight='bold', ha='center')
    ax.text(b1_x + 2.25, 3.6, "Moving-Block (B = 1,000)\nBlock 48 (24h) & Block 336 (7d)",
            color='#e0f2fe', fontsize=10, ha='center')

    b1_desc = (
        "• Resampling strictly within 16 continuous test segments\n"
        "• Zero cross-gap leakage over missing telemetry blackouts\n"
        "• Validates statistical significance of error reductions"
    )
    ax.text(b1_x + 2.25, 1.4, b1_desc, color='#e2e8f0', fontsize=9.5, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.5", fc='#1e293b', ec='#0284c7', lw=1.5))

    # Arrow between blocks
    ax.annotate("", xy=(7.2, 4.0), xytext=(6.3, 4.0),
                arrowprops=dict(arrowstyle="<->", color='#38bdf8', lw=2.5, mutation_scale=18))

    # Block 2: Multi-Metric Trade-Off
    b2_x = 7.5
    draw_isometric_block(ax, x=b2_x, y=2.8, w=4.5, h=2.5, depth=0.7,
                         color='#059669', top_color='#34d399', side_color='#047857',
                         alpha=0.85, edge_color='#ffffff')
    ax.text(b2_x + 2.25, 4.3, "METRIC TRADE-OFF ANALYSIS", color='#ffffff', fontsize=12, fontweight='bold', ha='center')
    ax.text(b2_x + 2.25, 3.6, "Objective Loss Asymmetry\nResidual KAN vs Persistence Naive",
            color='#d1fae5', fontsize=10, ha='center')

    b2_desc = (
        "• MAE: KAN Ensemble wins (519.78 vs 598.36 kW | -13.13%)\n"
        "• RMSE: KAN Ensemble wins (1019.95 vs 1160.44 kW | -12.11%)\n"
        "• sMAPE: Persistence Naive wins (13.35% vs 18.96%)\n"
        "• Conclusion: Models must be chosen based on dispatch penalty"
    )
    ax.text(b2_x + 2.25, 1.4, b2_desc, color='#e2e8f0', fontsize=9.5, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.5", fc='#1e293b', ec='#059669', lw=1.5))

    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "5_evaluation_3d.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print("Saved 5_evaluation_3d.png (Cleaned)")


if __name__ == "__main__":
    print("Regenerating clean 3D isometric architecture figures (Zero Overlap)...")
    make_fig1_data_pipeline()
    make_fig2_preprocessing_contract()
    make_fig3_model_benchmark()
    make_fig4_kan_ensemble()
    make_fig5_evaluation()
    print("All 5 clean figures regenerated successfully in docs/figures/!")
