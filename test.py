"""
analyze_and_recalculate_velocities.py
================================================================================
Scientific Velocity Analysis & In-Place Experimental Recalculation Engine
--------------------------------------------------------------------------------
1. Modifies the 4 existing experimental CSV files in place by adding:
      - RECALCULATED_VEL_X = DX_FROM_PREVIOUS_POINT / 10.0
      - RECALCULATED_VEL_Y = DY_FROM_PREVIOUS_POINT / 10.0
   (All other columns and existing values remain untouched).

2. Loads the 4 simulation datasets (filtered to FRAME % 6 == 0).

3. Compares THREE velocity distributions per regime in original physical units (µm/s):
      - Blue: Original Experimental (VEL_X, VEL_Y)
      - Green: Recalculated Experimental (RECALCULATED_VEL_X, RECALCULATED_VEL_Y)
      - Red: Simulation (VEL_X, VEL_Y)

4. Computes within-experiment consistency metrics (Mean Diff, MAD, Std Diff, Pearson r)
   and experimental-vs-simulation distribution metrics (Wasserstein, KS Stat).

5. Generates 4 dual-panel publication figures (300 DPI) and exports a master
   diagnostic summary table: velocity_recalculation_statistics.csv.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# =============================================================================
# 1. FILE PATHS & DIRECTORY CONFIGURATION
# =============================================================================
EXPERIMENTAL_KILLING_CANCER = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\Cyto_Cancer Cell Kinematics.csv"
)

EXPERIMENTAL_KILLING_TCELL = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\Cyto_T-Cell Kinematics.csv"
)

EXPERIMENTAL_NONKILLING_CANCER = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\Wt_Cancer Cell Kinematics.csv"
)

EXPERIMENTAL_NONKILLING_TCELL = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\Wt_T-Cell Kinematics.csv"
)

SIM_KILLING_CANCER = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\new_simulator\kinematic_csv_outputs"
    r"\killing_Cancer-cell_kinematics.csv"
)

SIM_KILLING_TCELL = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\new_simulator\kinematic_csv_outputs"
    r"\killing_T-cell_kinematics.csv"
)

SIM_NONKILLING_CANCER = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\new_simulator\kinematic_csv_outputs"
    r"\non-killing_Cancer-cell_kinematics.csv"
)

SIM_NONKILLING_TCELL = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\new_simulator\kinematic_csv_outputs"
    r"\non-killing_T-cell_kinematics.csv"
)

OUTPUT_DIR = r"velocity_recalculation_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMPARISON_CONFIGS = [
    {
        "name": "Killing Cancer",
        "exp_path": EXPERIMENTAL_KILLING_CANCER,
        "sim_path": SIM_KILLING_CANCER,
        "fig_name": "01_killing_cancer_velocity_comparison.png",
    },
    {
        "name": "Killing T-Cell",
        "exp_path": EXPERIMENTAL_KILLING_TCELL,
        "sim_path": SIM_KILLING_TCELL,
        "fig_name": "02_killing_tcell_velocity_comparison.png",
    },
    {
        "name": "Non-Killing Cancer",
        "exp_path": EXPERIMENTAL_NONKILLING_CANCER,
        "sim_path": SIM_NONKILLING_CANCER,
        "fig_name": "03_non_killing_cancer_velocity_comparison.png",
    },
    {
        "name": "Non-Killing T-Cell",
        "exp_path": EXPERIMENTAL_NONKILLING_TCELL,
        "sim_path": SIM_NONKILLING_TCELL,
        "fig_name": "04_non_killing_tcell_velocity_comparison.png",
    },
]


# =============================================================================
# 2. IN-PLACE EXPERIMENTAL CSV RECALCULATION
# =============================================================================
def recalculate_and_update_experimental_csv(filepath, description):
    """
    Loads experimental CSV, computes RECALCULATED_VEL_X and RECALCULATED_VEL_Y
    as (DX / 10.0) and (DY / 10.0), and writes directly back to the original CSV path.
    """
    print(f"\n[IN-PLACE UPDATE] Processing: {description}")
    print(f"  File Path: {filepath}")

    if not os.path.exists(filepath):
        print(f"  [ERROR] File missing: {filepath}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(filepath)

    if "DX_FROM_PREVIOUS_POINT" not in df.columns or "DY_FROM_PREVIOUS_POINT" not in df.columns:
        print(f"  [ERROR] Missing displacement columns in {filepath}", file=sys.stderr)
        sys.exit(1)

    # Strictly recalculate as displacement / 10.0 (no 1.6 scaling)
    dx_numeric = pd.to_numeric(df["DX_FROM_PREVIOUS_POINT"], errors="coerce")
    dy_numeric = pd.to_numeric(df["DY_FROM_PREVIOUS_POINT"], errors="coerce")

    df["RECALCULATED_VEL_X"] = dx_numeric / 10.0
    df["RECALCULATED_VEL_Y"] = dy_numeric / 10.0

    df.to_csv(filepath, index=False)
    print(f"  Successfully updated {len(df):,} rows in place with RECALCULATED_VEL_X and RECALCULATED_VEL_Y.")
    return df


# =============================================================================
# 3. SIMULATION DATA LOADER
# =============================================================================
def load_simulation_velocities(filepath, description, chunksize=200_000):
    """
    Loads simulation dataset in chunks, filters to observation frames (FRAME % 6 == 0),
    and returns clean velocity series.
    """
    print(f"\n[SIMULATION LOAD] Loading: {description}")
    print(f"  File Path: {filepath}")

    if not os.path.exists(filepath):
        print(f"  [ERROR] File missing: {filepath}", file=sys.stderr)
        sys.exit(1)

    chunks = []
    total_raw_rows = 0

    read_cols = ["VEL_X", "VEL_Y"]
    preview = pd.read_csv(filepath, nrows=2)
    if "FRAME" in preview.columns:
        read_cols.append("FRAME")

    for chunk in pd.read_csv(filepath, usecols=read_cols, chunksize=chunksize):
        total_raw_rows += len(chunk)
        if "FRAME" in chunk.columns:
            chunk = chunk[chunk["FRAME"] % 6 == 0]
        chunks.append(chunk[["VEL_X", "VEL_Y"]])

    sim_df = pd.concat(chunks, ignore_index=True)
    print(f"  Retained {len(sim_df):,} analysis frames (filtered from {total_raw_rows:,} total frames).")
    return sim_df


# =============================================================================
# 4. STATISTICAL UTILITIES
# =============================================================================
def clean_array(series):
    """Extracts finite 1D numeric numpy array from a pandas series."""
    arr = pd.to_numeric(series, errors="coerce").dropna().values
    return arr[np.isfinite(arr)]


def compute_descriptive_stats(arr, label):
    """Calculates comprehensive descriptive summary metrics for a velocity array."""
    if len(arr) == 0:
        return {
            "Label": label, "N": 0, "Mean": np.nan, "Std": np.nan,
            "Median": np.nan, "Min": np.nan, "Max": np.nan
        }
    return {
        "Label": label,
        "N": len(arr),
        "Mean": float(np.mean(arr)),
        "Std": float(np.std(arr)),
        "Median": float(np.median(arr)),
        "Min": float(np.min(arr)),
        "Max": float(np.max(arr)),
    }


def compute_internal_consistency(orig_series, recalc_series):
    """
    Computes paired consistency statistics between original and recalculated experimental velocities.
    """
    df_paired = pd.DataFrame({"orig": orig_series, "recalc": recalc_series}).dropna()
    df_paired = df_paired[np.isfinite(df_paired["orig"]) & np.isfinite(df_paired["recalc"])]

    if len(df_paired) == 0:
        return np.nan, np.nan, np.nan, np.nan

    diff = df_paired["recalc"] - df_paired["orig"]
    mean_diff = float(np.mean(diff))
    mad = float(np.mean(np.abs(diff)))
    std_diff = float(np.std(diff))

    if np.std(df_paired["orig"]) > 1e-12 and np.std(df_paired["recalc"]) > 1e-12:
        r, _ = stats.pearsonr(df_paired["orig"], df_paired["recalc"])
    else:
        r = np.nan

    return mean_diff, mad, std_diff, float(r)


# =============================================================================
# 5. DUAL-PANEL THREE-DISTRIBUTION PLOTTING ENGINE
# =============================================================================
def plot_three_distribution_velocity(exp_df, sim_df, title_prefix, fig_filename):
    """
    Generates a 2-panel figure (VEL_X on left, VEL_Y on right) containing:
      - Blue: Original Experimental
      - Green: Recalculated Experimental
      - Red: Simulation
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    components = [
        ("VEL_X", "RECALCULATED_VEL_X", "VEL_X", "Velocity X (µm/s)", axes[0]),
        ("VEL_Y", "RECALCULATED_VEL_Y", "VEL_Y", "Velocity Y (µm/s)", axes[1]),
    ]

    for orig_col, recalc_col, sim_col, xlabel, ax in components:
        exp_orig = clean_array(exp_df[orig_col])
        exp_recalc = clean_array(exp_df[recalc_col])
        sim_vals = clean_array(sim_df[sim_col])

        # Establish common support range across all 3 distributions (percentiles to suppress extreme outliers)
        all_vals = np.concatenate([exp_orig, exp_recalc, sim_vals])
        lo, hi = np.percentile(all_vals, [0.05, 99.95])
        bins = np.linspace(lo, hi, 61)

        # Plot 3 distributions
        ax.hist(
            exp_orig,
            bins=bins,
            density=True,
            alpha=0.45,
            color="#1f77b4",
            label="Original Experimental"
        )
        ax.hist(
            exp_recalc,
            bins=bins,
            density=True,
            alpha=0.45,
            color="#2ca02c",
            label="Recalculated Exp (DX/10)"
        )
        ax.hist(
            sim_vals,
            bins=bins,
            density=True,
            alpha=0.45,
            color="#d62728",
            label="Simulation"
        )

        ax.set_title(xlabel.split(" ")[0] + " " + xlabel.split(" ")[1], fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)

    fig.suptitle(
        f"{title_prefix} — Experimental Velocity Recalculation vs Simulation",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, fig_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED FIGURE] {output_path}")


# =============================================================================
# 6. MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    print("=========================================================================")
    print("   VELOCITY RECALCULATION & THREE-DISTRIBUTION COMPARISON PIPELINE       ")
    print("=========================================================================")
    print(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}\n")

    summary_records = []

    for cfg in COMPARISON_CONFIGS:
        regime = cfg["name"]
        print("\n" + "=" * 75)
        print(f"PROCESSING REGIME: {regime.upper()}")
        print("=" * 75)

        # Step 1: In-Place Update of Experimental Dataset
        exp_df = recalculate_and_update_experimental_csv(cfg["exp_path"], f"Experimental {regime}")

        # Step 2: Load Simulation Dataset (FRAME % 6 == 0)
        sim_df = load_simulation_velocities(cfg["sim_path"], f"Simulation {regime}")

        # Step 3: Extract and Clean 1D Velocity Arrays
        orig_vx = clean_array(exp_df["VEL_X"])
        recalc_vx = clean_array(exp_df["RECALCULATED_VEL_X"])
        sim_vx = clean_array(sim_df["VEL_X"])

        orig_vy = clean_array(exp_df["VEL_Y"])
        recalc_vy = clean_array(exp_df["RECALCULATED_VEL_Y"])
        sim_vy = clean_array(sim_df["VEL_Y"])

        # Step 4: Descriptive Statistics
        stats_orig_vx = compute_descriptive_stats(orig_vx, "Orig Exp VEL_X")
        stats_recalc_vx = compute_descriptive_stats(recalc_vx, "Recalc Exp VEL_X")
        stats_sim_vx = compute_descriptive_stats(sim_vx, "Sim VEL_X")

        stats_orig_vy = compute_descriptive_stats(orig_vy, "Orig Exp VEL_Y")
        stats_recalc_vy = compute_descriptive_stats(recalc_vy, "Recalc Exp VEL_Y")
        stats_sim_vy = compute_descriptive_stats(sim_vy, "Sim VEL_Y")

        print(f"\n--- Descriptive Statistics: VEL_X (µm/s) ---")
        print(f"  Orig Exp   : N={stats_orig_vx['N']:,} | Mean={stats_orig_vx['Mean']:+.4f} | Std={stats_orig_vx['Std']:.4f} | Med={stats_orig_vx['Median']:+.4f}")
        print(f"  Recalc Exp : N={stats_recalc_vx['N']:,} | Mean={stats_recalc_vx['Mean']:+.4f} | Std={stats_recalc_vx['Std']:.4f} | Med={stats_recalc_vx['Median']:+.4f}")
        print(f"  Simulation : N={stats_sim_vx['N']:,} | Mean={stats_sim_vx['Mean']:+.4f} | Std={stats_sim_vx['Std']:.4f} | Med={stats_sim_vx['Median']:+.4f}")

        print(f"\n--- Descriptive Statistics: VEL_Y (µm/s) ---")
        print(f"  Orig Exp   : N={stats_orig_vy['N']:,} | Mean={stats_orig_vy['Mean']:+.4f} | Std={stats_orig_vy['Std']:.4f} | Med={stats_orig_vy['Median']:+.4f}")
        print(f"  Recalc Exp : N={stats_recalc_vy['N']:,} | Mean={stats_recalc_vy['Mean']:+.4f} | Std={stats_recalc_vy['Std']:.4f} | Med={stats_recalc_vy['Median']:+.4f}")
        print(f"  Simulation : N={stats_sim_vy['N']:,} | Mean={stats_sim_vy['Mean']:+.4f} | Std={stats_sim_vy['Std']:.4f} | Med={stats_sim_vy['Median']:+.4f}")

        # Step 5: Internal Experimental Consistency Checks (Orig vs Recalc)
        mean_diff_x, mad_x, std_diff_x, r_x = compute_internal_consistency(exp_df["VEL_X"], exp_df["RECALCULATED_VEL_X"])
        mean_diff_y, mad_y, std_diff_y, r_y = compute_internal_consistency(exp_df["VEL_Y"], exp_df["RECALCULATED_VEL_Y"])

        print(f"\n--- Internal Experimental Consistency (Recalc vs Original) ---")
        print(f"  VEL_X: Mean Diff={mean_diff_x:+.4f} µm/s | MAD={mad_x:.4f} µm/s | Std Diff={std_diff_x:.4f} | Pearson r={r_x:.4f}")
        print(f"  VEL_Y: Mean Diff={mean_diff_y:+.4f} µm/s | MAD={mad_y:.4f} µm/s | Std Diff={std_diff_y:.4f} | Pearson r={r_y:.4f}")

        # Step 6: Distributional Comparison with Simulation
        w_orig_x = stats.wasserstein_distance(orig_vx, sim_vx)
        w_recalc_x = stats.wasserstein_distance(recalc_vx, sim_vx)
        ks_orig_x = stats.ks_2samp(orig_vx, sim_vx).statistic
        ks_recalc_x = stats.ks_2samp(recalc_vx, sim_vx).statistic

        w_orig_y = stats.wasserstein_distance(orig_vy, sim_vy)
        w_recalc_y = stats.wasserstein_distance(recalc_vy, sim_vy)
        ks_orig_y = stats.ks_2samp(orig_vy, sim_vy).statistic
        ks_recalc_y = stats.ks_2samp(recalc_vy, sim_vy).statistic

        print(f"\n--- Distributional Agreement with Simulation ---")
        print(f"  VEL_X  -> Orig vs Sim   : Wasserstein={w_orig_x:.4f} µm/s | KS Stat={ks_orig_x:.4f}")
        print(f"  VEL_X  -> Recalc vs Sim : Wasserstein={w_recalc_x:.4f} µm/s | KS Stat={ks_recalc_x:.4f}")
        print(f"  VEL_Y  -> Orig vs Sim   : Wasserstein={w_orig_y:.4f} µm/s | KS Stat={ks_orig_y:.4f}")
        print(f"  VEL_Y  -> Recalc vs Sim : Wasserstein={w_recalc_y:.4f} µm/s | KS Stat={ks_recalc_y:.4f}")

        # Step 7: Plot 3-Distribution Figures
        plot_three_distribution_velocity(exp_df, sim_df, regime, cfg["fig_name"])

        # Append to master records table
        for component, stats_o, stats_r, stats_s, m_diff, mad, s_diff, r_val, w_o, w_r, ks_o, ks_r in [
            ("VEL_X", stats_orig_vx, stats_recalc_vx, stats_sim_vx, mean_diff_x, mad_x, std_diff_x, r_x, w_orig_x, w_recalc_x, ks_orig_x, ks_recalc_x),
            ("VEL_Y", stats_orig_vy, stats_recalc_vy, stats_sim_vy, mean_diff_y, mad_y, std_diff_y, r_y, w_orig_y, w_recalc_y, ks_orig_y, ks_recalc_y),
        ]:
            summary_records.append({
                "Regime": regime,
                "Component": component,
                "Orig_Exp_N": stats_o["N"],
                "Orig_Exp_Mean": stats_o["Mean"],
                "Orig_Exp_Std": stats_o["Std"],
                "Orig_Exp_Median": stats_o["Median"],
                "Orig_Exp_Min": stats_o["Min"],
                "Orig_Exp_Max": stats_o["Max"],
                "Recalc_Exp_N": stats_r["N"],
                "Recalc_Exp_Mean": stats_r["Mean"],
                "Recalc_Exp_Std": stats_r["Std"],
                "Recalc_Exp_Median": stats_r["Median"],
                "Recalc_Exp_Min": stats_r["Min"],
                "Recalc_Exp_Max": stats_r["Max"],
                "Sim_N": stats_s["N"],
                "Sim_Mean": stats_s["Mean"],
                "Sim_Std": stats_s["Std"],
                "Sim_Median": stats_s["Median"],
                "Sim_Min": stats_s["Min"],
                "Sim_Max": stats_s["Max"],
                "Internal_Mean_Diff": m_diff,
                "Internal_MAD": mad,
                "Internal_Std_Diff": s_diff,
                "Internal_Pearson_r": r_val,
                "Wasserstein_Orig_vs_Sim": w_o,
                "Wasserstein_Recalc_vs_Sim": w_r,
                "KS_Orig_vs_Sim": ks_o,
                "KS_Recalc_vs_Sim": ks_r,
            })

    # Step 8: Export Master Summary CSV
    summary_df = pd.DataFrame(summary_records)
    csv_out_path = os.path.join(OUTPUT_DIR, "velocity_recalculation_statistics.csv")
    summary_df.to_csv(csv_out_path, index=False)

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETE")
    print(f"Master Summary CSV Exported to: {csv_out_path}")
    print(f"Figures Saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()