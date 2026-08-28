"""
plot_kinematic_distributions_raw.py
================================================================================
Scientific Kinematic Distribution Comparison Across 4 Temporal Windows
--------------------------------------------------------------------------------
Generates 16 publication-quality 17-panel comparison figures, neatly organized 
into 4 regime-specific subdirectories.

Temporal Horizons:
  1. Window 1 (0 - 5 hrs):   FRAME <= 300
  2. Window 2 (1 Day):       FRAME <= 1440
  3. Window 3 (Half Data):   FRAME <= Half of max frame length
  4. Window 4 (Full Run):    All frames (0 to Full)

Kinematic Features (17 total):
  - Positions, Displacements, Distances, Accumulated Distances (DX_ACC, DY_ACC): µm
  - Velocities, Speeds: µm/s
  - Accumulated Tracking Time (DT_ACC): s
  - Path Efficiency: dimensionless
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION & FILE PATHS (8 DISTINCT DATASETS)
# ============================================================

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

OUTPUT_DIR = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\kinematic_comparison_plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 17 KINEMATIC FEATURES & PHYSICAL UNITS
# ============================================================

KINEMATIC_COLUMNS = [
    "DX_FROM_PREVIOUS_POINT",
    "DY_FROM_PREVIOUS_POINT",
    "DISPLACEMENT_FROM_PREVIOUS_POINT",
    "VEL_X",
    "VEL_Y",
    "SPEED",
    "DX_FROM_ORIGIN",
    "DY_FROM_ORIGIN",
    "DISPLACEMENT_FROM_ORIGIN",
    "DX_ACC",
    "DY_ACC",
    "DT_ACC",
    "DISTANCE_TRAVELED",
    "PATH_EFFICIENCY",
    "AVERAGE_SPEED",
    "POSITION_X",
    "POSITION_Y",
]

FEATURE_UNITS = {
    "DX_FROM_PREVIOUS_POINT": "µm",
    "DY_FROM_PREVIOUS_POINT": "µm",
    "DISPLACEMENT_FROM_PREVIOUS_POINT": "µm",
    "VEL_X": "µm/s",
    "VEL_Y": "µm/s",
    "SPEED": "µm/s",
    "DX_FROM_ORIGIN": "µm",
    "DY_FROM_ORIGIN": "µm",
    "DISPLACEMENT_FROM_ORIGIN": "µm",
    "DX_ACC": "µm",
    "DY_ACC": "µm",
    "DT_ACC": "s",
    "DISTANCE_TRAVELED": "µm",
    "PATH_EFFICIENCY": "dimensionless",
    "AVERAGE_SPEED": "µm/s",
    "POSITION_X": "µm",
    "POSITION_Y": "µm",
}

NON_NEGATIVE_FEATURES = [
    "DISPLACEMENT_FROM_PREVIOUS_POINT",
    "DISPLACEMENT_FROM_ORIGIN",
    "DX_ACC",
    "DY_ACC",
    "DT_ACC",
    "DISTANCE_TRAVELED",
    "PATH_EFFICIENCY",
    "SPEED",
    "AVERAGE_SPEED",
]


# ============================================================
# MEMORY-EFFICIENT DATASET LOADER (RETAINS FRAME COLUMN)
# ============================================================

def load_dataset(path, name, is_simulation=False, chunksize=200_000):
    print("\n" + "=" * 70)
    print(f"Loading: {name}")
    print("=" * 70)
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"\nERROR: File does not exist:\n{path}")

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

        chunks.append(chunk)

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
# PLOT ONE COMPARISON IN RAW PHYSICAL UNITS (17 PANELS)
# ============================================================

def plot_comparison(
    experimental_df,
    simulation_df,
    experimental_name,
    simulation_name,
    time_window_label,
    output_dir,
    output_filename
):
    print("\n" + "-" * 70)
    print(f"GENERATING RAW PHYSICAL PLOT [{time_window_label}]")
    print(f"Experimental: {experimental_name} (N = {len(experimental_df):,})")
    print(f"Simulation:   {simulation_name} (N = {len(simulation_df):,})")
    print("-" * 70)

    n_features = len(KINEMATIC_COLUMNS)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))  # 6 rows x 3 cols grid (18 total slots)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 4.5 * nrows)
    )

    axes = np.asarray(axes).flatten()

    for i, feature in enumerate(KINEMATIC_COLUMNS):
        ax = axes[i]
        unit = FEATURE_UNITS[feature]

        exp_raw = clean_feature(experimental_df, feature)
        sim_raw = clean_feature(simulation_df, feature)

        if len(exp_raw) == 0 or len(sim_raw) == 0:
            ax.set_title(f"{feature}\n(no valid data)", fontsize=11, fontweight="bold")
            ax.axis("off")
            continue

        combined = np.concatenate([exp_raw, sim_raw])
        min_v, max_v = np.percentile(combined, [0.05, 99.95])

        # Experimental Distribution (Blue)
        ax.hist(
            exp_raw,
            bins=60,
            range=(min_v, max_v) if min_v < max_v else None,
            density=True,
            alpha=0.55,
            color="#1f77b4",
            label="Experimental"
        )

        # Simulation Distribution (Red)
        ax.hist(
            sim_raw,
            bins=60,
            range=(min_v, max_v) if min_v < max_v else None,
            density=True,
            alpha=0.55,
            color="#d62728",
            label="Simulation"
        )

        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Value ({unit})" if unit != "dimensionless" else "Value (dimensionless)", fontsize=9.5)
        ax.set_ylabel("Probability Density", fontsize=9.5)
        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    # Turn off unused 18th subplot
    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{experimental_name} vs {simulation_name}\n"
        f"Kinematic Distribution Comparison — {time_window_label} (Raw Physical Units)",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# MAIN EXECUTION PIPELINE (16 TOTAL COMPARISONS IN 4 DIRS)
# ============================================================

def main():
    print("\n" + "=" * 75)
    print("LOADING 8 MASTER DATASETS FOR TEMPORAL PARTITION ANALYSIS")
    print("=" * 75)

    datasets = {
        "exp_kill_cancer": load_dataset(EXPERIMENTAL_KILLING_CANCER, "Experimental Killing Cancer"),
        "exp_kill_tcell": load_dataset(EXPERIMENTAL_KILLING_TCELL, "Experimental Killing T-Cell"),
        "exp_nonkill_cancer": load_dataset(EXPERIMENTAL_NONKILLING_CANCER, "Experimental Non-Killing Cancer"),
        "exp_nonkill_tcell": load_dataset(EXPERIMENTAL_NONKILLING_TCELL, "Experimental Non-Killing T-Cell"),

        "sim_kill_cancer": load_dataset(SIM_KILLING_CANCER, "Simulated Killing Cancer", is_simulation=True),
        "sim_kill_tcell": load_dataset(SIM_KILLING_TCELL, "Simulated Killing T-Cell", is_simulation=True),
        "sim_nonkill_cancer": load_dataset(SIM_NONKILLING_CANCER, "Simulated Non-Killing Cancer", is_simulation=True),
        "sim_nonkill_tcell": load_dataset(SIM_NONKILLING_TCELL, "Simulated Non-Killing T-Cell", is_simulation=True),
    }

    regimes = [
        ("Killing Cancer", "exp_kill_cancer", "sim_kill_cancer", "killing_cancer"),
        ("Non-Killing Cancer", "exp_nonkill_cancer", "sim_nonkill_cancer", "nonkilling_cancer"),
        ("Killing T-Cell", "exp_kill_tcell", "sim_kill_tcell", "killing_tcell"),
        ("Non-Killing T-Cell", "exp_nonkill_tcell", "sim_nonkill_tcell", "nonkilling_tcell"),
    ]

    print("\n" + "=" * 75)
    print("GENERATING 16 TEMPORAL REGIME COMPARISON FIGURES (ORGANIZED INTO 4 FOLDERS)")
    print("=" * 75)

    image_counter = 1

    for reg_title, exp_key, sim_key, reg_file_tag in regimes:
        exp_df_full = datasets[exp_key]
        sim_df_full = datasets[sim_key]

        # Create regime-specific sub-directory
        regime_dir_name = reg_title.replace(" ", "_").replace("-", "_")
        regime_out_dir = os.path.join(OUTPUT_DIR, regime_dir_name)
        os.makedirs(regime_out_dir, exist_ok=True)

        # Determine dynamic half-point frame for this specific pair
        max_exp_f = exp_df_full["FRAME"].max() if "FRAME" in exp_df_full.columns else 0
        max_sim_f = sim_df_full["FRAME"].max() if "FRAME" in sim_df_full.columns else 0
        max_frame_observed = int(max(max_exp_f, max_sim_f))
        half_frame_cutoff = max_frame_observed // 2

        time_windows = [
            ("Time Horizon 0-5 hrs (Frames 0-300)", 300, "01_0to5hrs"),
            ("Time Horizon 1 Day (Frames 0-1440)", 1440, "02_1day"),
            (f"Time Horizon Half Data (Frames 0-{half_frame_cutoff})", half_frame_cutoff, "03_half_data"),
            (f"Time Horizon Full Data (Frames 0-{max_frame_observed})", None, "04_full_data"),
        ]

        for time_label, frame_cutoff, time_file_tag in time_windows:
            # Apply Temporal Cutoff
            if frame_cutoff is not None and "FRAME" in exp_df_full.columns:
                exp_sub = exp_df_full[exp_df_full["FRAME"] <= frame_cutoff].copy()
            else:
                exp_sub = exp_df_full.copy()

            if frame_cutoff is not None and "FRAME" in sim_df_full.columns:
                sim_sub = sim_df_full[sim_df_full["FRAME"] <= frame_cutoff].copy()
            else:
                sim_sub = sim_df_full.copy()

            out_filename = f"{image_counter:02d}_{reg_file_tag}_{time_file_tag}.png"

            plot_comparison(
                exp_sub,
                sim_sub,
                f"Experimental {reg_title}",
                f"Simulated {reg_title}",
                time_label,
                regime_out_dir,
                out_filename
            )

            image_counter += 1

    print("\n" + "=" * 75)
    print("ALL 16 PUBLICATION FIGURES GENERATED AND SAVED")
    print("=" * 75)
    print(f"Master Output Directory: {os.path.abspath(OUTPUT_DIR)}\n")


if __name__ == "__main__":
    main()