"""
plot_kinematic_distributions_raw.py
================================================================================
Scientific Kinematic Distribution Comparison in Native Physical Units
--------------------------------------------------------------------------------
Generates 4 publication-quality 14-panel comparison figures (one per regime).
Compares experimental live-cell tracking datasets directly against simulation
trajectories sampled at observation intervals (FRAME % 6 == 0).

Plots distributions in raw physical measurement units:
  - Positions, Displacements, Distances: µm
  - Velocities, Speeds: µm/s
  - Path Efficiency: dimensionless
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION & FILE PATHS (8 DISTINCT DATASETS)
# ============================================================

# Experimental Datasets
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

# Simulation Datasets
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

# Output Directory
OUTPUT_DIR = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\kinematic_comparison_plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 14 KINEMATIC FEATURES & PHYSICAL UNITS
# ============================================================

KINEMATIC_COLUMNS = [
    "POSITION_X",
    "POSITION_Y",
    "DX_FROM_PREVIOUS_POINT",
    "DY_FROM_PREVIOUS_POINT",
    "DISPLACEMENT_FROM_PREVIOUS_POINT",
    "DX_FROM_ORIGIN",
    "DY_FROM_ORIGIN",
    "DISPLACEMENT_FROM_ORIGIN",
    "DISTANCE_TRAVELED",
    "PATH_EFFICIENCY",
    "VEL_X",
    "VEL_Y",
    "SPEED",
    "AVERAGE_SPEED",
]

FEATURE_UNITS = {
    "POSITION_X": "µm",
    "POSITION_Y": "µm",
    "DX_FROM_PREVIOUS_POINT": "µm",
    "DY_FROM_PREVIOUS_POINT": "µm",
    "DISPLACEMENT_FROM_PREVIOUS_POINT": "µm",
    "DX_FROM_ORIGIN": "µm",
    "DY_FROM_ORIGIN": "µm",
    "DISPLACEMENT_FROM_ORIGIN": "µm",
    "DISTANCE_TRAVELED": "µm",
    "PATH_EFFICIENCY": "dimensionless",
    "VEL_X": "µm/s",
    "VEL_Y": "µm/s",
    "SPEED": "µm/s",
    "AVERAGE_SPEED": "µm/s",
}

NON_NEGATIVE_FEATURES = [
    "DISPLACEMENT_FROM_PREVIOUS_POINT",
    "DISPLACEMENT_FROM_ORIGIN",
    "DISTANCE_TRAVELED",
    "PATH_EFFICIENCY",
    "SPEED",
    "AVERAGE_SPEED",
]


# ============================================================
# MEMORY-EFFICIENT DATASET LOADER
# ============================================================

def load_dataset(path, name, is_simulation=False, chunksize=200_000):
    print("\n" + "=" * 70)
    print(f"Loading: {name}")
    print("=" * 70)
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"\nERROR: File does not exist:\n{path}")

    # Read preview to identify available columns
    preview = pd.read_csv(path, nrows=2)
    cols_present = [col for col in KINEMATIC_COLUMNS if col in preview.columns]
    
    missing = [col for col in KINEMATIC_COLUMNS if col not in preview.columns]
    if missing:
        raise ValueError(
            f"\n{name} is missing the following kinematic columns:\n"
            + "\n".join(missing)
        )

    read_cols = cols_present + (["FRAME"] if "FRAME" in preview.columns else [])
    
    chunks = []
    total_raw_rows = 0

    for chunk in pd.read_csv(path, usecols=read_cols, chunksize=chunksize):
        total_raw_rows += len(chunk)
        
        # Simulation post-sampling: retain observation frames (FRAME % 6 == 0)
        if is_simulation and "FRAME" in chunk.columns:
            chunk = chunk[chunk["FRAME"] % 6 == 0]
            
        chunks.append(chunk[cols_present])

    df = pd.concat(chunks, ignore_index=True)

    filter_info = f" (filtered from {total_raw_rows:,} full frames)" if is_simulation else ""
    print(f"Retained Analysis Rows: {len(df):,}{filter_info}")
    print(f"Columns Verified: {len(df.columns)}")

    return df


# ============================================================
# FEATURE CLEANING (NO RESCALING)
# ============================================================

def clean_feature(df, feature):
    values = pd.to_numeric(df[feature], errors="coerce").dropna().values
    valid_mask = np.isfinite(values)

    if feature in NON_NEGATIVE_FEATURES:
        valid_mask = valid_mask & (values >= 0)

    return values[valid_mask]


# ============================================================
# PLOT ONE COMPARISON IN RAW PHYSICAL UNITS
# ============================================================

def plot_comparison(
    experimental_df,
    simulation_df,
    experimental_name,
    simulation_name,
    output_filename
):
    print("\n" + "-" * 70)
    print(f"GENERATING RAW PHYSICAL PLOT")
    print(f"Experimental: {experimental_name}")
    print(f"Simulation:   {simulation_name}")
    print("-" * 70)

    n_features = len(KINEMATIC_COLUMNS)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 4.8 * nrows)
    )

    axes = np.asarray(axes).flatten()

    for i, feature in enumerate(KINEMATIC_COLUMNS):
        ax = axes[i]
        unit = FEATURE_UNITS[feature]

        # Extract raw physical values
        exp_raw = clean_feature(experimental_df, feature)
        sim_raw = clean_feature(simulation_df, feature)

        if len(exp_raw) == 0 or len(sim_raw) == 0:
            ax.set_title(f"{feature}\n(no valid data)")
            ax.axis("off")
            continue

        # Common support range for direct histogram comparison
        combined = np.concatenate([exp_raw, sim_raw])
        min_v, max_v = np.percentile(combined, [0.05, 99.95])  # robust against outliers

        # Direct Raw Physical Histograms
        ax.hist(
            exp_raw,
            bins=60,
            range=(min_v, max_v) if min_v < max_v else None,
            density=True,
            alpha=0.55,
            color="#1f77b4",
            label="Experimental"
        )

        ax.hist(
            sim_raw,
            bins=60,
            range=(min_v, max_v) if min_v < max_v else None,
            density=True,
            alpha=0.55,
            color="#d62728",
            label="Simulation"
        )

        # Labels & Units
        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Value ({unit})" if unit != "dimensionless" else "Value (dimensionless)", fontsize=9.5)
        ax.set_ylabel("Probability Density", fontsize=9.5)
        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    # Turn off unused subplot (panel 15)
    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{experimental_name} vs {simulation_name}\n"
        "Kinematic Distribution Comparison (Raw Physical Units)",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("LOADING ALL 8 DATASETS FOR DIRECT PHYSICAL COMPARISON")
    print("=" * 70)

    # 1. Load Experimental Datasets (Pre-sampled at 6-frame observation intervals)
    exp_kill_cancer = load_dataset(EXPERIMENTAL_KILLING_CANCER, "Experimental Killing Cancer")
    exp_kill_tcell = load_dataset(EXPERIMENTAL_KILLING_TCELL, "Experimental Killing T-Cell")
    exp_nonkill_cancer = load_dataset(EXPERIMENTAL_NONKILLING_CANCER, "Experimental Non-Killing Cancer")
    exp_nonkill_tcell = load_dataset(EXPERIMENTAL_NONKILLING_TCELL, "Experimental Non-Killing T-Cell")

    # 2. Load Simulation Datasets (Filters to FRAME % 6 == 0 during loading)
    sim_kill_cancer = load_dataset(SIM_KILLING_CANCER, "Simulated Killing Cancer", is_simulation=True)
    sim_kill_tcell = load_dataset(SIM_KILLING_TCELL, "Simulated Killing T-Cell", is_simulation=True)
    sim_nonkill_cancer = load_dataset(SIM_NONKILLING_CANCER, "Simulated Non-Killing Cancer", is_simulation=True)
    sim_nonkill_tcell = load_dataset(SIM_NONKILLING_TCELL, "Simulated Non-Killing T-Cell", is_simulation=True)

    print("\n" + "=" * 70)
    print("GENERATING FOUR ONE-TO-ONE REGIME COMPARISON PLOTS")
    print("=" * 70)

    # Comparison 1: Killing Cancer
    plot_comparison(
        exp_kill_cancer,
        sim_kill_cancer,
        "Experimental Killing Cancer",
        "Simulated Killing Cancer",
        "01_experimental_vs_simulated_killing_cancer.png"
    )

    # Comparison 2: Non-Killing Cancer
    plot_comparison(
        exp_nonkill_cancer,
        sim_nonkill_cancer,
        "Experimental Non-Killing Cancer",
        "Simulated Non-Killing Cancer",
        "02_experimental_vs_simulated_nonkilling_cancer.png"
    )

    # Comparison 3: Killing T-Cell
    plot_comparison(
        exp_kill_tcell,
        sim_kill_tcell,
        "Experimental Killing T-Cell",
        "Simulated Killing T-Cell",
        "03_experimental_vs_simulated_killing_tcell.png"
    )

    # Comparison 4: Non-Killing T-Cell
    plot_comparison(
        exp_nonkill_tcell,
        sim_nonkill_tcell,
        "Experimental Non-Killing T-Cell",
        "Simulated Non-Killing T-Cell",
        "04_experimental_vs_simulated_nonkilling_tcell.png"
    )

    print("\n" + "=" * 70)
    print("ALL COMPARISONS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Output Figures Saved to:\n{OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()