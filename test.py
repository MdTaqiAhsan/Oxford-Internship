"""
plot_tcell_regime_comparison.py
================================================================================
Comparative Kinematic Distribution Analysis for T-Cells (Killing vs Non-Killing)
--------------------------------------------------------------------------------
Plots 3 side-by-side distribution panels for T-cells:
  1. Average Speed (AVERAGE_SPEED in µm/s)
  2. Max Reach / Distance Traveled (DISTANCE_TRAVELED in µm)
  3. Path Efficiency (PATH_EFFICIENCY, dimensionless)

Color Coding:
  - Red:   Cyto / Killing (Ct)
  - Green: Wild-Type / Non-Killing (WT)
================================================================================
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# FILE PATHS
# =============================================================================
CYTO_TCELL_FILE = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\Cyto_T-Cell Kinematics.csv"
)

WT_TCELL_FILE = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\Wt_T-Cell Kinematics.csv"
)

OUTPUT_DIR = r"tcell_regime_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADER & CLEANER
# =============================================================================
def load_tcell_metrics(filepath, label):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    cols = ["AVERAGE_SPEED", "DISTANCE_TRAVELED", "PATH_EFFICIENCY"]
    df = pd.read_csv(filepath, usecols=cols)

    clean_dict = {}
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        vals = vals[np.isfinite(vals)]
        if col in ["AVERAGE_SPEED", "DISTANCE_TRAVELED"]:
            vals = vals[vals >= 0]
        elif col == "PATH_EFFICIENCY":
            vals = vals[(vals >= 0.0) & (vals <= 1.0)]
        clean_dict[col] = vals

    print(f"Loaded {label}: {len(df):,} rows")
    return clean_dict


# =============================================================================
# PLOTTING
# =============================================================================
def main():
    print("Loading T-cell datasets...")
    cyto_data = load_tcell_metrics(CYTO_TCELL_FILE, "Cyto (Killing / Ct)")
    wt_data = load_tcell_metrics(WT_TCELL_FILE, "WT (Non-Killing / WT)")

    features = [
        ("AVERAGE_SPEED", "Average Speed (µm/s)", "µm/s"),
        ("DISTANCE_TRAVELED", "Max Reach / Distance Traveled (µm)", "µm"),
        ("PATH_EFFICIENCY", "Path Efficiency", "dimensionless"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    for i, (feat_key, title_label, unit) in enumerate(features):
        ax = axes[i]

        ct_vals = cyto_data[feat_key]
        wt_vals = wt_data[feat_key]

        # Common binning support
        combined = np.concatenate([ct_vals, wt_vals])
        if feat_key == "PATH_EFFICIENCY":
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(combined, [0.05, 99.5])

        bins = np.linspace(lo, hi, 50)

        # Plot WT (Non-Killing) in Green
        ax.hist(
            wt_vals,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#2ca02c",
            label="WT (Non-Killing)",
        )

        # Plot Ct (Killing) in Red
        ax.hist(
            ct_vals,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#d62728",
            label="Ct (Killing)",
        )

        ax.set_title(title_label, fontsize=12, fontweight="bold")
        ax.set_xlabel(
            f"Value ({unit})" if unit != "dimensionless" else "Value (0 to 1)",
            fontsize=10.5,
        )
        ax.set_ylabel("Probability Density", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)

    fig.suptitle(
        "T-Cell Kinematic Comparison: Killing (Ct) vs Non-Killing (WT)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "tcell_killing_vs_nonkilling_kinematics.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()