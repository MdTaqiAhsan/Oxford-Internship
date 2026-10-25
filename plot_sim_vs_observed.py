"""
plot_sim_vs_observed.py
================================================================================
Scientific Kinematic Distribution Comparison Across 2 Temporal Windows
--------------------------------------------------------------------------------
Generates 8 publication-quality 17-panel comparison figures across 4 regimes:
  1. Window 1 (0 - 5 hrs):   FRAME <= 1800 (18,000 s)
  2. Window 2 (Full / 24h):  All frames (0 to Full Dataset, ~8635 frames)

Sampling Logic:
  - Experimental (Observed): Every row taken as-is (rows exist every 6th frame).
  - Simulation: Subsampled at `FRAME % 6 == 0` to align observation intervals.

Filtering:
  - Toggleable trajectory-duration filter:
      Cancer: max(DT_ACC) >= 900 s
      Immune: max(DT_ACC) >= 3600 s

Usage:
    python plot_sim_vs_observed.py                       # latest run in results/simulations/
    python plot_sim_vs_observed.py --sim_dir <folder>    # a specific simulation run
Figures go to results/comparisons/<simulation run>/plots/.
================================================================================
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt

import project_paths as pp

# ============================================================
# CONFIGURATION & FILE PATHS (8 DISTINCT DATASETS)
# ============================================================

EXPERIMENTAL_KILLING_CANCER = pp.EXPERIMENTAL_DATA["killing"]["cancer"]
EXPERIMENTAL_KILLING_TCELL = pp.EXPERIMENTAL_DATA["killing"]["tcell"]
EXPERIMENTAL_NONKILLING_CANCER = pp.EXPERIMENTAL_DATA["non-killing"]["cancer"]
EXPERIMENTAL_NONKILLING_TCELL = pp.EXPERIMENTAL_DATA["non-killing"]["tcell"]

# Simulation datasets and the output folder depend on the chosen simulation
# run; they are set in main().
SIM_KILLING_CANCER = SIM_KILLING_TCELL = SIM_NONKILLING_CANCER = SIM_NONKILLING_TCELL = None
OUTPUT_DIR = None

# ------------------------------------------------------------
# TOGGLES & CONSTRAINTS
# ------------------------------------------------------------
ENABLE_TRAJECTORY_DURATION_FILTER = True

CANCER_MIN_DT_ACC_SECONDS = 900.0   # 15 min (~90 raw frames * 10s)
IMMUNE_MIN_DT_ACC_SECONDS = 3600.0  # 1 hour (~360 raw frames * 10s)

# Stride applied strictly to simulation data
SIM_OBSERVATION_STRIDE = 6  # Subsample condition: FRAME % 6 == 0

# Temporal boundary for 5 hours: 5 hrs * 3600 s/hr / 10 s/frame = 1800
FIVE_HOUR_FRAME_CUTOFF = 1800

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
# DATASET LOADER
# ============================================================

def load_dataset(path, name, is_simulation=False, is_cancer=False, chunksize=200_000):
    print("\n" + "=" * 75)
    print(f"Loading: {name}")
    print("=" * 75)
    print(f"Path: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"\nERROR: File does not exist:\n{path}")

    preview = pd.read_csv(path, nrows=2)
    cols_present = [col for col in KINEMATIC_COLUMNS if col in preview.columns]

    missing = [col for col in KINEMATIC_COLUMNS if col not in preview.columns]
    if missing:
        raise ValueError(f"\n{name} is missing columns:\n" + "\n".join(missing))

    # Identify primary track ID column
    track_col = None
    for cand in ["TRACK_ID_TA", "TRACK_ID"]:
        if cand in preview.columns:
            track_col = cand
            break

    min_dur = CANCER_MIN_DT_ACC_SECONDS if is_cancer else IMMUNE_MIN_DT_ACC_SECONDS

    # Pass 1: Trajectory duration filtering (if enabled)
    valid_tracks = None
    if ENABLE_TRAJECTORY_DURATION_FILTER and track_col is not None and "DT_ACC" in preview.columns:
        print(f"Scanning trajectories for duration threshold (>= {min_dur:.0f}s)...")
        track_max = {}
        for chunk in pd.read_csv(path, usecols=[track_col, "DT_ACC"], chunksize=chunksize):
            grp = chunk.groupby(track_col)["DT_ACC"].max()
            for tid, val in grp.items():
                if pd.notnull(val):
                    track_max[tid] = max(track_max.get(tid, 0.0), float(val))

        valid_tracks = {tid for tid, val in track_max.items() if val >= min_dur}
        print(f"  • Retained Trajectories: {len(valid_tracks):,} / {len(track_max):,} "
              f"({(len(valid_tracks)/len(track_max)*100.0 if len(track_max) > 0 else 0):.2f}%)")
    elif not ENABLE_TRAJECTORY_DURATION_FILTER:
        print("Trajectory duration filtering is DISABLED. Retaining all tracks.")

    # Pass 2: Data Extraction & Selective Subsampling
    read_cols = cols_present + (["FRAME"] if "FRAME" in preview.columns else [])
    if track_col and track_col not in read_cols:
        read_cols.append(track_col)

    chunks = []
    total_raw_rows = 0

    for chunk in pd.read_csv(path, usecols=read_cols, chunksize=chunksize):
        total_raw_rows += len(chunk)

        # Retain only qualifying trajectories if filter is active
        if valid_tracks is not None and track_col in chunk.columns:
            chunk = chunk[chunk[track_col].isin(valid_tracks)]

        # Apply 6-frame stride exclusively to simulation data
        if is_simulation and "FRAME" in chunk.columns and SIM_OBSERVATION_STRIDE > 1:
            chunk = chunk[chunk["FRAME"] % SIM_OBSERVATION_STRIDE == 0]

        chunks.append(chunk[cols_present + (["FRAME"] if "FRAME" in chunk.columns else [])])

    df = pd.concat(chunks, ignore_index=True)
    sampling_info = f" (subsampled 1:{SIM_OBSERVATION_STRIDE} from {total_raw_rows:,} rows)" if is_simulation else f" (all {total_raw_rows:,} rows retained)"
    print(f"Loaded Analysis Rows: {len(df):,}{sampling_info}")
    return df


# ============================================================
# FEATURE CLEANING & VALIDATION
# ============================================================

def clean_feature(df, feature):
    values = pd.to_numeric(df[feature], errors="coerce").dropna().values
    valid_mask = np.isfinite(values)

    if feature in NON_NEGATIVE_FEATURES:
        valid_mask = valid_mask & (values >= 0.0)
    if feature == "PATH_EFFICIENCY":
        valid_mask = valid_mask & (values <= 1.05)

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
    print(f"  -> Generating: {output_filename} [{time_window_label}]")

    n_features = len(KINEMATIC_COLUMNS)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows))
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

        if not np.isfinite(min_v) or not np.isfinite(max_v) or min_v == max_v:
            plot_range = (min_v - 1.0, max_v + 1.0)
        else:
            plot_range = (min_v, max_v)

        ax.hist(
            exp_raw,
            bins=60,
            range=plot_range,
            density=True,
            alpha=0.55,
            color="#1f77b4",
            label="Experimental (Every 6th Frame)"
        )

        ax.hist(
            sim_raw,
            bins=60,
            range=plot_range,
            density=True,
            alpha=0.55,
            color="#d62728",
            label=f"Simulation (Every {SIM_OBSERVATION_STRIDE}th Frame)"
        )

        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Value ({unit})" if unit != "dimensionless" else "Value (dimensionless)", fontsize=9.5)
        ax.set_ylabel("Probability Density", fontsize=9.5)
        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

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


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    global SIM_KILLING_CANCER, SIM_KILLING_TCELL, SIM_NONKILLING_CANCER, SIM_NONKILLING_TCELL, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Plot experimental vs simulated kinematic distributions.")
    parser.add_argument("--sim_dir", default=None,
                        help="Simulation run folder (default: latest run in results/simulations/)")
    args = parser.parse_args()

    sim_dir = os.path.abspath(args.sim_dir) if args.sim_dir else pp.latest_run_dir(pp.SIMULATIONS_DIR)
    SIM_KILLING_CANCER = pp.simulation_csv_path(sim_dir, "killing", "cancer")
    SIM_KILLING_TCELL = pp.simulation_csv_path(sim_dir, "killing", "tcell")
    SIM_NONKILLING_CANCER = pp.simulation_csv_path(sim_dir, "non-killing", "cancer")
    SIM_NONKILLING_TCELL = pp.simulation_csv_path(sim_dir, "non-killing", "tcell")
    OUTPUT_DIR = os.path.join(pp.COMPARISONS_DIR, os.path.basename(os.path.normpath(sim_dir)), "plots")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Simulation run: {sim_dir}")

    print("\n" + "=" * 75)
    print("LOADING MASTER DATASETS (OBSERVED: ALL FRAMES | SIM: EVERY 6TH FRAME)")
    print("=" * 75)

    datasets = {
        # Experimental datasets: is_simulation=False -> no frame striding applied
        "exp_kill_cancer": load_dataset(EXPERIMENTAL_KILLING_CANCER, "Experimental Killing Cancer", is_simulation=False, is_cancer=True),
        "exp_kill_tcell": load_dataset(EXPERIMENTAL_KILLING_TCELL, "Experimental Killing T-Cell", is_simulation=False, is_cancer=False),
        "exp_nonkill_cancer": load_dataset(EXPERIMENTAL_NONKILLING_CANCER, "Experimental Non-Killing Cancer", is_simulation=False, is_cancer=True),
        "exp_nonkill_tcell": load_dataset(EXPERIMENTAL_NONKILLING_TCELL, "Experimental Non-Killing T-Cell", is_simulation=False, is_cancer=False),

        # Simulation datasets: is_simulation=True -> subsampled at FRAME % 6 == 0
        "sim_kill_cancer": load_dataset(SIM_KILLING_CANCER, "Simulated Killing Cancer", is_simulation=True, is_cancer=True),
        "sim_kill_tcell": load_dataset(SIM_KILLING_TCELL, "Simulated Killing T-Cell", is_simulation=True, is_cancer=False),
        "sim_nonkill_cancer": load_dataset(SIM_NONKILLING_CANCER, "Simulated Non-Killing Cancer", is_simulation=True, is_cancer=True),
        "sim_nonkill_tcell": load_dataset(SIM_NONKILLING_TCELL, "Simulated Non-Killing T-Cell", is_simulation=True, is_cancer=False),
    }

    regimes = [
        ("Killing Cancer", "exp_kill_cancer", "sim_kill_cancer", "killing_cancer"),
        ("Non-Killing Cancer", "exp_nonkill_cancer", "sim_nonkill_cancer", "nonkilling_cancer"),
        ("Killing T-Cell", "exp_kill_tcell", "sim_kill_tcell", "killing_tcell"),
        ("Non-Killing T-Cell", "exp_nonkill_tcell", "sim_nonkill_tcell", "nonkilling_tcell"),
    ]

    print("\n" + "=" * 75)
    print("GENERATING 8 REGIME COMPARISON FIGURES (2 PER REGIME)")
    print("=" * 75)

    image_counter = 1

    for reg_title, exp_key, sim_key, reg_file_tag in regimes:
        exp_df_full = datasets[exp_key]
        sim_df_full = datasets[sim_key]

        regime_dir_name = reg_title.replace(" ", "_").replace("-", "_")
        regime_out_dir = os.path.join(OUTPUT_DIR, regime_dir_name)
        os.makedirs(regime_out_dir, exist_ok=True)

        max_exp_f = exp_df_full["FRAME"].max() if "FRAME" in exp_df_full.columns else 0
        max_sim_f = sim_df_full["FRAME"].max() if "FRAME" in sim_df_full.columns else 0
        max_frame_observed = int(max(max_exp_f, max_sim_f))

        time_windows = [
            ("Time Horizon 0-5 hrs (Frames 0-1800)", FIVE_HOUR_FRAME_CUTOFF, "01_0to5hrs"),
            (f"Time Horizon Full 24 hrs (Frames 0-{max_frame_observed})", None, "02_full_data"),
        ]

        for time_label, frame_cutoff, time_file_tag in time_windows:
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
    print("ALL 8 PUBLICATION FIGURES GENERATED AND SAVED")
    print(f"Master Output Directory: {os.path.abspath(OUTPUT_DIR)}\n")


if __name__ == "__main__":
    main()