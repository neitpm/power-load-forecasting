# -*- coding: utf-8 -*-
"""
=============================================================================
VISUALIZE 3D ARCHITECTURE & PIPELINE FIGURES (300 DPI)
For Scientific ML & Industrial Time-Series Forecasting Portfolio
=============================================================================
Outputs generated in docs/figures/:
1. 1_data_pipeline_3d.png
2. 2_preprocessing_contract_3d.png
3. 3_model_benchmark_3d.png
4. 4_kan_ensemble_3d.png
5. 5_evaluation_3d.png
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches

# High DPI and styling configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = False

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_3d_box(ax, x, y, z, dx, dy, dz, color='#2563eb', alpha=0.8, edge_color='#1e293b'):
    """Draw a 3D isometric cuboid block."""
    xx = [x, x+dx, x+dx, x, x, x+dx, x+dx, x]
    yy = [y, y, y+dy, y+dy, y, y, y+dy, y+dy]
    zz = [z, z, z, z, z+dz, z+dz, z+dz, z+dz]
    vertices = [
        [[xx[0], yy[0], zz[0]], [xx[1], yy[1], zz[0]], [xx[2], yy[2], zz[0]], [xx[3], yy[3], zz[0]]], # bottom
        [[xx[4], yy[4], zz[4]], [xx[5], yy[5], zz[4]], [xx[6], yy[6], zz[4]], [xx[7], yy[7], zz[4]]], # top
        [[xx[0], yy[0], zz[0]], [xx[1], yy[1], zz[0]], [xx[5], yy[5], zz[4]], [xx[4], yy[4], zz[4]]], # front
        [[xx[2], yy[2], zz[0]], [xx[3], yy[3], zz[0]], [xx[7], yy[7], zz[4]], [xx[6], yy[6], zz[4]]], # back
        [[xx[0], yy[0], zz[0]], [xx[3], yy[3], zz[0]], [xx[7], yy[7], zz[4]], [xx[4], yy[4], zz[4]]], # left
        [[xx[1], yy[1], zz[0]], [xx[2], yy[2], zz[0]], [xx[6], yy[6], zz[4]], [xx[5], yy[5], zz[4]]]  # right
    ]
    poly = Poly3DCollection(vertices, facecolors=color, linewidths=1.2, edgecolors=edge_color, alpha=alpha)
    ax.add_collection3d(poly)


# -----------------------------------------------------------------------------
# Figure 1: 1_data_pipeline_3d.png
# -----------------------------------------------------------------------------
def make_fig1_data_pipeline():
    fig = plt.figure(figsize=(12, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')

    # Draw 4 volumetric processing stages
    stages = [
        ("1. Industrial 3-Phase\nRaw Metering\n(U, I, cosφ, P, Q)", 0, '#0284c7'),
        ("2. Timestamp Alignment\n& 30-min Step Regularization", 3, '#0d9488'),
        ("3. Outlier Filter &\nGap Partitioning", 6, '#d97706'),
        ("4. 16 Continuous Segments\n(14,385 Test Records)", 9, '#16a34a')
    ]

    for title, x_pos, col in stages:
        draw_3d_box(ax, x=x_pos, y=1, z=0, dx=2, dy=4, dz=2, color=col, alpha=0.75, edge_color='#ffffff')
        ax.text(x_pos + 1.0, 3.0, 2.4, title, color='#ffffff', fontsize=9, fontweight='bold',
                ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", fc='#1e293b', ec=col, lw=1))

    # Connect with 3D arrows
    for x_arr in [2.2, 5.2, 8.2]:
        ax.quiver(x_arr, 3.0, 1.0, 0.6, 0, 0, color='#38bdf8', arrow_length_ratio=0.3, linewidth=2.5)

    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 7)
    ax.set_zlim(-0.5, 4.5)
    ax.view_init(elev=24, azim=-55)
    ax.axis('off')
    plt.title("Industrial Telemetry Pipeline & Continuous Segment Extraction\nFeeder 480 - Substation 224 Tan Hung (Tay Ninh Power Co.)",
              color='#f8fafc', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_data_pipeline_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved 1_data_pipeline_3d.png")


# -----------------------------------------------------------------------------
# Figure 2: 2_preprocessing_contract_3d.png
# -----------------------------------------------------------------------------
def make_fig2_preprocessing_contract():
    fig = plt.figure(figsize=(12, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0b0f19')
    fig.patch.set_facecolor('#0b0f19')

    # Draw chronological splits with isolation barrier
    draw_3d_box(ax, x=0, y=0, z=0, dx=5, dy=3, dz=1.8, color='#3b82f6', alpha=0.8, edge_color='#93c5fd')
    ax.text(2.5, 1.5, 2.1, "TRAIN SPLIT\n03/2020 - 01/2024\n(Fit Scaler & Imputer)",
            color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc='#1e3a8a', ec='#60a5fa'))

    draw_3d_box(ax, x=6, y=0, z=0, dx=2.2, dy=3, dz=1.8, color='#f59e0b', alpha=0.8, edge_color='#fde68a')
    ax.text(7.1, 1.5, 2.1, "VALIDATION SPLIT\n01/2024 - 11/2024\n(Early Stopping Only)",
            color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc='#78350f', ec='#fbbf24'))

    draw_3d_box(ax, x=9, y=0, z=0, dx=3.2, dy=3, dz=1.8, color='#10b981', alpha=0.8, edge_color='#a7f3d0')
    ax.text(10.6, 1.5, 2.1, "TEST SPLIT (14,385 records)\n11/2024 - 08/2025\n(Strict Zero Leakage)",
            color='#ffffff', fontsize=9, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc='#064e3b', ec='#34d399'))

    # Isolation barriers
    draw_3d_box(ax, x=5.3, y=-0.5, z=0, dx=0.4, dy=4, dz=2.5, color='#ef4444', alpha=0.4, edge_color='#f87171')
    ax.text(5.5, 2, 3.0, "BARRIER 1:\nNo Lookahead", color='#fca5a5', fontsize=8, ha='center', fontweight='bold')

    draw_3d_box(ax, x=8.4, y=-0.5, z=0, dx=0.4, dy=4, dz=2.5, color='#ef4444', alpha=0.4, edge_color='#f87171')
    ax.text(8.6, 2, 3.0, "BARRIER 2:\nZero Test Tuning", color='#fca5a5', fontsize=8, ha='center', fontweight='bold')

    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 5)
    ax.set_zlim(0, 4)
    ax.view_init(elev=22, azim=-50)
    ax.axis('off')
    plt.title("Strict Chronological Data Splitting & Zero Data Leakage Contract",
              color='#f8fafc', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_preprocessing_contract_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved 2_preprocessing_contract_3d.png")


# -----------------------------------------------------------------------------
# Figure 3: 3_model_benchmark_3d.png
# -----------------------------------------------------------------------------
def make_fig3_model_benchmark():
    fig = plt.figure(figsize=(12, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')

    # Models and metrics
    models = ['Persistence\n(Naive)', 'Moving\nAverage', 'Ridge', 'XGBoost', 'Residual KAN\n(Ensemble 5s)']
    mae_vals = [598.36, 1016.16, 730.96, 645.24, 519.78]
    rmse_vals = [1160.44, 1586.70, 1186.89, 1155.21, 1019.95]
    colors = ['#64748b', '#94a3b8', '#38bdf8', '#3b82f6', '#10b981']

    x_base = np.arange(len(models)) * 2.2

    for i in range(len(models)):
        # Draw MAE bar (normalized height for 3D visual)
        h_mae = mae_vals[i] / 400.0
        draw_3d_box(ax, x=x_base[i], y=1, z=0, dx=0.8, dy=0.8, dz=h_mae, color=colors[i], alpha=0.85)
        ax.text(x_base[i] + 0.4, 1.4, h_mae + 0.1, f"{mae_vals[i]:.1f}\nkW",
                color='#ffffff', fontsize=8, ha='center', va='bottom', fontweight='bold')

        # Draw RMSE bar
        h_rmse = rmse_vals[i] / 400.0
        draw_3d_box(ax, x=x_base[i] + 0.9, y=1, z=0, dx=0.8, dy=0.8, dz=h_rmse, color=colors[i], alpha=0.55)
        ax.text(x_base[i] + 1.3, 1.4, h_rmse + 0.1, f"{rmse_vals[i]:.1f}\nkW",
                color='#cbd5e1', fontsize=8, ha='center', va='bottom')

        # Model label
        ax.text(x_base[i] + 0.8, 0.4, 0, models[i], color='#f1f5f9', fontsize=8.5, ha='center', va='top', fontweight='bold')

    # Legend annotations
    ax.text(0, 3.2, 3.8, "■ Solid: MAE (kW)   |   ▨ Shaded: RMSE (kW)", color='#38bdf8', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc='#1e293b', ec='#475569'))
    ax.text(8.8, 3.2, 3.8, "Residual KAN Ensemble:\nΔMAE: -13.13%\nΔRMSE: -12.11%", color='#34d399', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc='#064e3b', ec='#10b981'))

    ax.set_xlim(-1, 11.5)
    ax.set_ylim(0, 4)
    ax.set_zlim(0, 4.8)
    ax.view_init(elev=20, azim=-55)
    ax.axis('off')
    plt.title("Error Metric Comparison Across Benchmarked Families (30-min Horizon)",
              color='#f8fafc', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_model_benchmark_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved 3_model_benchmark_3d.png")


# -----------------------------------------------------------------------------
# Figure 4: 4_kan_ensemble_3d.png
# -----------------------------------------------------------------------------
def make_fig4_kan_ensemble():
    fig = plt.figure(figsize=(12, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0b0f19')
    fig.patch.set_facecolor('#0b0f19')

    seeds = [42, 101, 2024, 7, 999]
    y_offsets = [0, 1.8, 3.6, 5.4, 7.2]

    # Draw 5 individual seed model blocks
    for i, (seed, y_pos) in enumerate(zip(seeds, y_offsets)):
        draw_3d_box(ax, x=1, y=y_pos, z=0, dx=2.5, dy=1.2, dz=1.2, color='#6366f1', alpha=0.8, edge_color='#a5b4fc')
        ax.text(2.25, y_pos + 0.6, 1.4, f"Seed {seed}\nResidual KAN", color='#ffffff', fontsize=8.5,
                ha='center', va='bottom', fontweight='bold')
        # Arrow towards averaging block
        ax.quiver(3.7, y_pos + 0.6, 0.6, 2.0, (3.6 - y_pos)*0.25, 0, color='#818cf8', arrow_length_ratio=0.25, linewidth=2)

    # Central Uniform Ensemble Block
    draw_3d_box(ax, x=6.5, y=2.6, z=-0.2, dx=3.5, dy=3.2, dz=1.8, color='#10b981', alpha=0.9, edge_color='#6ee7b7')
    ax.text(8.25, 4.2, 2.0, "Uniform 5-Seed Ensemble\ny_ens = (1/5) * Σ y_i\nZero Test Weight Tuning",
            color='#ffffff', fontsize=10, ha='center', va='bottom', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc='#064e3b', ec='#34d399'))

    # Final output arrow
    ax.quiver(10.2, 4.2, 0.7, 1.8, 0, 0, color='#34d399', arrow_length_ratio=0.3, linewidth=2.5)
    ax.text(12.2, 4.2, 0.7, "Final Forecast\nP_pred (kW)", color='#a7f3d0', fontsize=9.5, fontweight='bold', va='center')

    ax.set_xlim(0, 13)
    ax.set_ylim(-1, 9)
    ax.set_zlim(-0.5, 3.5)
    ax.view_init(elev=26, azim=-60)
    ax.axis('off')
    plt.title("Multi-Seed Residual KAN Architecture & Uniform Ensembling Pipeline",
              color='#f8fafc', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_kan_ensemble_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved 4_kan_ensemble_3d.png")


# -----------------------------------------------------------------------------
# Figure 5: 5_evaluation_3d.png
# -----------------------------------------------------------------------------
def make_fig5_evaluation():
    fig = plt.figure(figsize=(12, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')

    # Draw evaluation cube representing Contiguous Moving-Block Bootstrap
    draw_3d_box(ax, x=1, y=1, z=0, dx=4, dy=3, dz=2, color='#0284c7', alpha=0.7, edge_color='#7dd3fc')
    ax.text(3, 2.5, 2.3, "Contiguous Moving-Block\nBootstrap (B=1000)\nBlock 48 & Block 336",
            color='#ffffff', fontsize=9.5, ha='center', va='bottom', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc='#0c4a6e', ec='#38bdf8'))

    # Draw Multi-metric trade-off space
    draw_3d_box(ax, x=7, y=1, z=0, dx=4, dy=3, dz=2, color='#059669', alpha=0.7, edge_color='#6ee7b7')
    ax.text(9, 2.5, 2.3, "Multi-Metric Evaluation:\n• MAE / RMSE (KAN Wins)\n• sMAPE (Persistence Wins)\n• Strict Integrity",
            color='#ffffff', fontsize=9.5, ha='center', va='bottom', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc='#064e3b', ec='#34d399'))

    ax.quiver(5.2, 2.5, 1.0, 1.6, 0, 0, color='#38bdf8', arrow_length_ratio=0.3, linewidth=2.5)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 3.5)
    ax.view_init(elev=22, azim=-50)
    ax.axis('off')
    plt.title("Statistical Significance Validation & Multi-Metric Trade-off Verification",
              color='#f8fafc', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "5_evaluation_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved 5_evaluation_3d.png")


if __name__ == "__main__":
    print("Generating 3D architecture figures...")
    make_fig1_data_pipeline()
    make_fig2_preprocessing_contract()
    make_fig3_model_benchmark()
    make_fig4_kan_ensemble()
    make_fig5_evaluation()
    print("All 5 figures generated successfully at docs/figures/!")
