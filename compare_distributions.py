"""
compare_distributions.py
================================================================================
Scientific Quantitative Distribution Comparison Analysis for Cell Kinematics
Focused Analysis: 0 to 5 Hours (FRAME <= 1800) with Trajectory Duration Filtering
--------------------------------------------------------------------------------
Compares 4 regime-matched experimental datasets against 4 simulation datasets.

Filtering Pipeline:
  1. Trajectory Duration Filtering:
     - Cancer Cells: exclude entire track if total track duration < 900 s (15 min)
     - Immune/T-Cells: exclude entire track if total track duration < 3600 s (1 hr)
  2. Temporal Window:
     - Retains only observations within the 0 to 5-hour window (FRAME <= 1800).
  3. Simulation Subsampling:
     - Retains observational frames (FRAME % 6 == 0) from the continuous simulation.

Evaluates 17 Kinematic Features across:
  - Primary Analysis: Raw Physical Units (Wasserstein, KS, JSD)
  - Secondary Diagnostic Analysis: Experimental Reference Standardized Z-Scores

Usage:
    python compare_distributions.py                       # latest run in results/simulations/
    python compare_distributions.py --sim_dir <folder>    # a specific simulation run
Outputs go to results/comparisons/<simulation run>/distribution_metrics_0to5hrs/.
================================================================================
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import project_paths as pp

# =============================================================================
# 1. CONFIGURATION & FILE PATHS (8 DATASETS)
# =============================================================================
FRAME_MAX_CUTOFF = 1800  # 0 to 5 hours (1800 frames * 10s = 18000s = 5 hr context)

# Track duration thresholds. DT_ACC is in seconds in both the extracted
# experimental files and the simulation output.
CANCER_MIN_SECONDS = 900.0    # 15 minutes minimum tracking duration
IMMUNE_MIN_SECONDS = 3600.0   # 1 hour minimum tracking duration

# Experimental Datasets
EXPERIMENTAL_KILLING_CANCER = pp.EXPERIMENTAL_DATA["killing"]["cancer"]
EXPERIMENTAL_KILLING_TCELL = pp.EXPERIMENTAL_DATA["killing"]["tcell"]
EXPERIMENTAL_NONKILLING_CANCER = pp.EXPERIMENTAL_DATA["non-killing"]["cancer"]
EXPERIMENTAL_NONKILLING_TCELL = pp.EXPERIMENTAL_DATA["non-killing"]["tcell"]

# Simulation datasets and the output folder depend on the chosen simulation
# run; they are set in main().
KILLING_CANCER = KILLING_TCELL = NONKILLING_CANCER = NONKILLING_TCELL = None
OUTPUT_DIR = None

# =============================================================================
# 2. FEATURE DEFINITIONS, PHYSICAL UNITS, CATEGORIES & COMPARISONS
# =============================================================================
FEATURES = [
    "POSITION_X",
    "POSITION_Y",
    "DX_FROM_PREVIOUS_POINT",
    "DY_FROM_PREVIOUS_POINT",
    "DISPLACEMENT_FROM_PREVIOUS_POINT",
    "DX_FROM_ORIGIN",
    "DY_FROM_ORIGIN",
    "DISPLACEMENT_FROM_ORIGIN",
    "DX_ACC",
    "DY_ACC",
    "DT_ACC",
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
    "DX_ACC": "µm",
    "DY_ACC": "µm",
    "DT_ACC": "s",
    "DISTANCE_TRAVELED": "µm",
    "PATH_EFFICIENCY": "dimensionless",
    "VEL_X": "µm/s",
    "VEL_Y": "µm/s",
    "SPEED": "µm/s",
    "AVERAGE_SPEED": "µm/s",
}

CATEGORIES = {
    "Spatial Behavior": [
        "POSITION_X",
        "POSITION_Y",
        "DX_FROM_ORIGIN",
        "DY_FROM_ORIGIN",
        "DISPLACEMENT_FROM_ORIGIN",
    ],
    "Local Motility": [
        "DX_FROM_PREVIOUS_POINT",
        "DY_FROM_PREVIOUS_POINT",
        "DISPLACEMENT_FROM_PREVIOUS_POINT",
    ],
    "Velocity": [
        "VEL_X",
        "VEL_Y",
        "SPEED",
    ],
    "Accumulated Motility": [
        "DX_ACC",
        "DY_ACC",
        "DT_ACC",
    ],
    "Long-Term Motility": [
        "DISTANCE_TRAVELED",
        "PATH_EFFICIENCY",
        "AVERAGE_SPEED",
    ],
}

COMPARISONS = [
    (
        "Experimental Killing Cancer vs Killing Cancer",
        "Experimental Killing Cancer",
        "Killing Cancer",
    ),
    (
        "Experimental Non-Killing Cancer vs Non-Killing Cancer",
        "Experimental Non-Killing Cancer",
        "Non-Killing Cancer",
    ),
    (
        "Experimental Killing T-Cell vs Killing T-Cell",
        "Experimental Killing T-Cell",
        "Killing T-Cell",
    ),
    (
        "Experimental Non-Killing T-Cell vs Non-Killing T-Cell",
        "Experimental Non-Killing T-Cell",
        "Non-Killing T-Cell",
    ),
]

COMPARISON_NAMES = [c[0] for c in COMPARISONS]

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


# =============================================================================
# 3. MATHEMATICAL METRIC IMPLEMENTATIONS
# =============================================================================
def calculate_wasserstein(arr1, arr2):
    return float(stats.wasserstein_distance(arr1, arr2))


def calculate_ks_statistic(arr1, arr2):
    res = stats.ks_2samp(arr1, arr2)
    return float(res.statistic), float(res.pvalue)


def calculate_jsd_standardized(arr1_z, arr2_z, z_min=-10.0, z_max=10.0, num_bins=200):
    clamped1 = np.clip(arr1_z, z_min, z_max)
    clamped2 = np.clip(arr2_z, z_min, z_max)
    bins = np.linspace(z_min, z_max, num_bins + 1)

    p_hist, _ = np.histogram(clamped1, bins=bins, density=False)
    q_hist, _ = np.histogram(clamped2, bins=bins, density=False)

    p = p_hist / np.sum(p_hist) if np.sum(p_hist) > 0 else p_hist
    q = q_hist / np.sum(q_hist) if np.sum(q_hist) > 0 else q_hist

    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    jsd = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(jsd, 0.0, 1.0))


def calculate_jsd_raw(arr1, arr2, num_bins=100):
    combined = np.concatenate([arr1, arr2])
    min_val, max_val = np.min(combined), np.max(combined)

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, num_bins + 1)

    p_hist, _ = np.histogram(arr1, bins=bins, density=False)
    q_hist, _ = np.histogram(arr2, bins=bins, density=False)

    p = p_hist / np.sum(p_hist) if np.sum(p_hist) > 0 else p_hist
    q = q_hist / np.sum(q_hist) if np.sum(q_hist) > 0 else q_hist

    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    jsd = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(jsd, 0.0, 1.0))


# =============================================================================
# 4. DATA LOADING & TWO-STAGE FILTERING (DURATION + 0-5 HRS)
# =============================================================================
def load_and_validate_dataset(
    filepath,
    description,
    is_simulation=False,
    is_cancer=False,
    chunksize=200_000
):
    print(f"\n -> Loading [{description}] from:\n    {filepath}")
    if not os.path.exists(filepath):
        print(f" [ERROR] File missing: {filepath}", file=sys.stderr)
        sys.exit(1)

    min_track_duration = CANCER_MIN_SECONDS if is_cancer else IMMUNE_MIN_SECONDS
    cell_type_str = "Cancer (>= 900 s)" if is_cancer else "Immune/T-Cell (>= 3600 s)"
    print(f"    Duration Filter Threshold: {cell_type_str}")

    preview = pd.read_csv(filepath, nrows=2)

    # Determine track ID column
    track_col = None
    for cand in ["TRACK_ID_TA", "TRACK_ID", "ID", "TRACKID", "CELL_ID"]:
        if cand in preview.columns:
            track_col = cand
            break

    # --- PASS 1: Calculate full trajectory durations per track (DT_ACC, seconds) ---
    track_durations = {}
    if track_col is not None and "DT_ACC" in preview.columns:
        for chunk in pd.read_csv(filepath, usecols=[track_col, "DT_ACC"], chunksize=chunksize):
            seconds = pd.to_numeric(chunk["DT_ACC"], errors="coerce")
            grp = seconds.groupby(chunk[track_col]).max()
            for t_id, max_v in grp.items():
                if pd.notnull(max_v):
                    track_durations[t_id] = max(track_durations.get(t_id, 0), max_v)

        valid_tracks = {
            t_id for t_id, dur in track_durations.items() if dur >= min_track_duration
        }
        print(
            f"    [Pass 1] Total Tracks: {len(track_durations):,} | "
            f"Retained (Dur >= {min_track_duration:.0f} s): {len(valid_tracks):,}"
        )
    else:
        valid_tracks = None

    # --- PASS 2: Filter to valid tracks and FRAME <= 1800 ---
    cols_present = [f for f in FEATURES if f in preview.columns]
    read_cols = list(set(cols_present + (["FRAME"] if "FRAME" in preview.columns else []) + ([track_col] if track_col else [])))

    feature_data = {f: [] for f in FEATURES}
    metadata = {
        "total_rows": 0,
        "sampled_rows": 0,
        "unique_tracks": set(),
        "invalid_counts": {f: 0 for f in FEATURES},
        "path_eff_out_of_bounds": 0,
    }

    for chunk in pd.read_csv(filepath, usecols=read_cols, chunksize=chunksize):
        metadata["total_rows"] += len(chunk)

        # 1. Filter tracks by duration threshold
        if valid_tracks is not None and track_col in chunk.columns:
            chunk = chunk[chunk[track_col].isin(valid_tracks)]

        # 2. Simulation post-sampling (FRAME % 6 == 0)
        if is_simulation and "FRAME" in chunk.columns:
            chunk = chunk[chunk["FRAME"] % 6 == 0]

        # 3. Filter to 0 to 5 hours (FRAME <= 1800)
        if "FRAME" in chunk.columns:
            chunk = chunk[chunk["FRAME"] <= FRAME_MAX_CUTOFF]

        if len(chunk) == 0:
            continue

        metadata["sampled_rows"] += len(chunk)
        if track_col and track_col in chunk.columns:
            metadata["unique_tracks"].update(chunk[track_col].dropna().unique())

        for feat in FEATURES:
            if feat in chunk.columns:
                s = pd.to_numeric(chunk[feat], errors="coerce")
                valid_mask = s.notnull() & np.isfinite(s)

                if feat in NON_NEGATIVE_FEATURES:
                    valid_mask = valid_mask & (s >= 0)

                if feat == "PATH_EFFICIENCY":
                    oob_mask = valid_mask & ((s < -1e-5) | (s > 1.05))
                    metadata["path_eff_out_of_bounds"] += oob_mask.sum()

                metadata["invalid_counts"][feat] += len(s) - valid_mask.sum()
                feature_data[feat].append(s[valid_mask].values)

    clean_data = {}
    for feat in FEATURES:
        if len(feature_data[feat]) > 0:
            clean_data[feat] = np.concatenate(feature_data[feat])
        else:
            clean_data[feat] = np.array([], dtype=np.float64)

    num_tracks = len(metadata["unique_tracks"])
    obs_per_track = (
        metadata["sampled_rows"] / num_tracks if num_tracks > 0 else 0.0
    )

    print(
        f"    [Pass 2] Final Analysis Observations (0-5 hrs): {metadata['sampled_rows']:,} | "
        f"{num_tracks:,} unique tracks | ~{obs_per_track:.1f} obs/track"
    )

    metadata["obs_per_track"] = obs_per_track
    return clean_data, metadata


# =============================================================================
# 5. HEATMAP ENGINE
# =============================================================================
def generate_heatmap(df, title, cbar_label, filename, output_dir, cmap="YlOrRd", fmt=".3f"):
    plt.figure(figsize=(12, 10), dpi=300)
    sns.set_theme(style="white")

    ax = sns.heatmap(
        df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.75,
        linecolor="#E0E0E0",
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        annot_kws={"size": 9, "weight": "bold"},
    )

    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Kinematic Feature", fontsize=11, fontweight="bold")
    plt.xlabel("Regime Comparison", fontsize=11, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Saved Heatmap: {filepath}")


# =============================================================================
# 6. MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    global KILLING_CANCER, KILLING_TCELL, NONKILLING_CANCER, NONKILLING_TCELL, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Compare experimental and simulated kinematic distributions (0-5 h).")
    parser.add_argument("--sim_dir", default=None,
                        help="Simulation run folder (default: latest run in results/simulations/)")
    args = parser.parse_args()

    sim_dir = os.path.abspath(args.sim_dir) if args.sim_dir else pp.latest_run_dir(pp.SIMULATIONS_DIR)
    KILLING_CANCER = pp.simulation_csv_path(sim_dir, "killing", "cancer")
    KILLING_TCELL = pp.simulation_csv_path(sim_dir, "killing", "tcell")
    NONKILLING_CANCER = pp.simulation_csv_path(sim_dir, "non-killing", "cancer")
    NONKILLING_TCELL = pp.simulation_csv_path(sim_dir, "non-killing", "tcell")
    OUTPUT_DIR = os.path.join(pp.COMPARISONS_DIR, os.path.basename(os.path.normpath(sim_dir)),
                              "distribution_metrics_0to5hrs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Simulation run: {sim_dir}")

    print("=========================================================================")
    print("   QUANTITATIVE KINEMATIC ANALYSIS PIPELINE (0 TO 5 HOURS ONLY)          ")
    print("=========================================================================")
    print(f"Temporal Window: FRAME <= {FRAME_MAX_CUTOFF} (0 - 5 hours)")
    print(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}\n")

    # Step 1: Load All 8 Datasets with Thresholds
    print("--- STEP 1: LOADING & FILTERING ALL 8 DATASETS ---")
    datasets = {}
    metadata_summary = {}

    dataset_configs = [
        ("Experimental Killing Cancer", EXPERIMENTAL_KILLING_CANCER, False, True),
        ("Experimental Non-Killing Cancer", EXPERIMENTAL_NONKILLING_CANCER, False, True),
        ("Experimental Killing T-Cell", EXPERIMENTAL_KILLING_TCELL, False, False),
        ("Experimental Non-Killing T-Cell", EXPERIMENTAL_NONKILLING_TCELL, False, False),
        ("Killing Cancer", KILLING_CANCER, True, True),
        ("Killing T-Cell", KILLING_TCELL, True, False),
        ("Non-Killing Cancer", NONKILLING_CANCER, True, True),
        ("Non-Killing T-Cell", NONKILLING_TCELL, True, False),
    ]

    for name, path, is_sim, is_can in dataset_configs:
        data, meta = load_and_validate_dataset(
            path, name, is_simulation=is_sim, is_cancer=is_can
        )
        datasets[name] = data
        metadata_summary[name] = meta

    # Step 2: Compute Raw and Standardized Metrics
    print("\n--- STEP 2: CALCULATING METRICS FOR 0 TO 5 HOURS ---")

    raw_wass_df = pd.DataFrame(index=FEATURES, columns=COMPARISON_NAMES, dtype=float)
    raw_ks_df = pd.DataFrame(index=FEATURES, columns=COMPARISON_NAMES, dtype=float)
    raw_ks_p_df = pd.DataFrame(index=FEATURES, columns=COMPARISON_NAMES, dtype=float)
    raw_jsd_df = pd.DataFrame(index=FEATURES, columns=COMPARISON_NAMES, dtype=float)

    std_wass_df = pd.DataFrame(index=FEATURES, columns=COMPARISON_NAMES, dtype=float)
    std_ks_df = pd.DataFrame(index=FEATURES, columns=COMPARISON_NAMES, dtype=float)
    std_jsd_df = pd.DataFrame(index=FEATURES, columns=COMPARISON_NAMES, dtype=float)

    feature_stats_rows = []

    for comp_name, exp_key, sim_key in COMPARISONS:
        print(f"\nProcessing Comparison: [{comp_name}] ({exp_key} vs {sim_key})...")

        exp_data = datasets[exp_key]
        sim_data = datasets[sim_key]

        for feat in FEATURES:
            e_arr = exp_data[feat]
            s_arr = sim_data[feat]

            if len(e_arr) == 0 or len(s_arr) == 0:
                print(f" [WARNING] Empty array for feature {feat} in {comp_name}")
                continue

            # Summary Statistics
            e_mean, e_std = float(np.mean(e_arr)), float(np.std(e_arr))
            s_mean, s_std = float(np.mean(s_arr)), float(np.std(s_arr))
            e_median, e_iqr = float(np.median(e_arr)), float(stats.iqr(e_arr))
            s_median, s_iqr = float(np.median(s_arr)), float(stats.iqr(s_arr))

            # Raw Metrics
            w_raw = calculate_wasserstein(e_arr, s_arr)
            ks_stat_raw, ks_p_raw = calculate_ks_statistic(e_arr, s_arr)
            jsd_raw = calculate_jsd_raw(e_arr, s_arr)

            raw_wass_df.loc[feat, comp_name] = w_raw
            raw_ks_df.loc[feat, comp_name] = ks_stat_raw
            raw_ks_p_df.loc[feat, comp_name] = ks_p_raw
            raw_jsd_df.loc[feat, comp_name] = jsd_raw

            # Experimental Reference Standardization
            if e_std > 1e-12:
                e_std_arr = (e_arr - e_mean) / e_std
                s_std_arr = (s_arr - e_mean) / e_std
                delta_z = float((s_mean - e_mean) / e_std)
            else:
                e_std_arr = e_arr - e_mean
                s_std_arr = s_arr - e_mean
                delta_z = float(s_mean - e_mean)

            w_std = calculate_wasserstein(e_std_arr, s_std_arr)
            ks_stat_std, _ = calculate_ks_statistic(e_std_arr, s_std_arr)
            jsd_std = calculate_jsd_standardized(e_std_arr, s_std_arr)

            std_wass_df.loc[feat, comp_name] = w_std
            std_ks_df.loc[feat, comp_name] = ks_stat_std
            std_jsd_df.loc[feat, comp_name] = jsd_std

            feature_stats_rows.append({
                "Comparison": comp_name,
                "Feature": feat,
                "Units": FEATURE_UNITS[feat],
                "Exp_Mean": e_mean,
                "Exp_Std": e_std,
                "Exp_Median": e_median,
                "Exp_IQR": e_iqr,
                "Sim_Mean": s_mean,
                "Sim_Std": s_std,
                "Sim_Median": s_median,
                "Sim_IQR": s_iqr,
                "Delta_Z": delta_z,
                "Raw_Wasserstein": w_raw,
                "Raw_KS": ks_stat_raw,
                "Raw_JSD": jsd_raw,
                "Std_Wasserstein": w_std,
                "Std_KS": ks_stat_std,
                "Std_JSD": jsd_std,
            })

    feature_stats_df = pd.DataFrame(feature_stats_rows)

    # Step 3: Composite Difference Scores
    print("\n--- STEP 3: COMPUTING STANDARDIZED DISTRIBUTIONAL DIFFERENCE SCORES ---")

    def min_max_norm_metric(df):
        min_v = df.values.min()
        max_v = df.values.max()
        if max_v == min_v:
            return pd.DataFrame(0.0, index=df.index, columns=df.columns)
        return (df - min_v) / (max_v - min_v)

    norm_wass = min_max_norm_metric(std_wass_df)
    norm_ks = min_max_norm_metric(std_ks_df)
    norm_jsd = min_max_norm_metric(std_jsd_df)

    composite_df = (1.0 / 3.0) * norm_wass + (1.0 / 3.0) * norm_ks + (1.0 / 3.0) * norm_jsd

    category_df = pd.DataFrame(index=list(CATEGORIES.keys()), columns=COMPARISON_NAMES, dtype=float)
    for cat_name, cat_feats in CATEGORIES.items():
        category_df.loc[cat_name] = composite_df.loc[cat_feats].mean(axis=0)

    overall_comp_series = category_df.mean(axis=0).sort_values(ascending=True)
    comp_summary_df = pd.DataFrame({
        "Comparison": overall_comp_series.index,
        "Overall_Composite_Score": overall_comp_series.values,
        "Spatial_Behavior": category_df.loc["Spatial Behavior", overall_comp_series.index].values,
        "Local_Motility": category_df.loc["Local Motility", overall_comp_series.index].values,
        "Velocity": category_df.loc["Velocity", overall_comp_series.index].values,
        "Accumulated_Motility": category_df.loc["Accumulated Motility", overall_comp_series.index].values,
        "Long_Term_Motility": category_df.loc["Long-Term Motility", overall_comp_series.index].values,
    })
    comp_summary_df["Rank"] = np.arange(1, len(comp_summary_df) + 1)

    # Step 4: Rankings per Comparison
    rankings_dict = {}
    for comp in COMPARISON_NAMES:
        sorted_feats = composite_df[comp].sort_values(ascending=True)
        comp_rank_df = pd.DataFrame({
            "Feature": sorted_feats.index,
            "Units": [FEATURE_UNITS[f] for f in sorted_feats.index],
            "Composite_Score": sorted_feats.values,
            "Std_Wasserstein": std_wass_df.loc[sorted_feats.index, comp].values,
            "Std_KS": std_ks_df.loc[sorted_feats.index, comp].values,
            "Std_JSD": std_jsd_df.loc[sorted_feats.index, comp].values,
            "Raw_Wasserstein": raw_wass_df.loc[sorted_feats.index, comp].values,
            "Raw_KS": raw_ks_df.loc[sorted_feats.index, comp].values,
            "Raw_JSD": raw_jsd_df.loc[sorted_feats.index, comp].values,
        })
        comp_rank_df["Rank"] = np.arange(1, len(sorted_feats) + 1)
        rankings_dict[comp] = comp_rank_df

    # Step 5: Export CSV Tables
    print("\n--- STEP 5: SAVING CSV METRIC TABLES ---")
    raw_wass_df.to_csv(os.path.join(OUTPUT_DIR, "raw_wasserstein.csv"))
    raw_ks_df.to_csv(os.path.join(OUTPUT_DIR, "raw_ks.csv"))
    raw_jsd_df.to_csv(os.path.join(OUTPUT_DIR, "raw_jsd.csv"))

    std_wass_df.to_csv(os.path.join(OUTPUT_DIR, "standardized_wasserstein.csv"))
    std_ks_df.to_csv(os.path.join(OUTPUT_DIR, "standardized_ks.csv"))
    std_jsd_df.to_csv(os.path.join(OUTPUT_DIR, "standardized_jsd.csv"))

    composite_df.to_csv(os.path.join(OUTPUT_DIR, "composite_scores.csv"))
    category_df.to_csv(os.path.join(OUTPUT_DIR, "category_scores.csv"))
    feature_stats_df.to_csv(os.path.join(OUTPUT_DIR, "feature_statistics.csv"), index=False)
    comp_summary_df.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"), index=False)

    all_ranks_list = []
    for comp, df_r in rankings_dict.items():
        df_temp = df_r.copy()
        df_temp["Comparison"] = comp
        all_ranks_list.append(df_temp)
    pd.concat(all_ranks_list, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "feature_rankings.csv"), index=False
    )
    print(" -> All CSV metric tables exported successfully.")

    # Step 6: Generate Publication Heatmaps
    print("\n--- STEP 6: GENERATING PUBLICATION HEATMAP FIGURES ---")
    generate_heatmap(
        std_wass_df,
        "Wasserstein Distance (0-5 hrs, Experimental Standardized Units)",
        "Wasserstein Distance (σ_exp units)",
        "wasserstein_heatmap.png",
        OUTPUT_DIR,
    )
    generate_heatmap(
        std_ks_df,
        "Kolmogorov-Smirnov Statistic (0-5 hrs, Descriptive CDF Disagreement)",
        "KS Statistic (0 to 1)",
        "ks_heatmap.png",
        OUTPUT_DIR,
    )
    generate_heatmap(
        std_jsd_df,
        "Jensen-Shannon Divergence (0-5 hrs, Standardized Z-Space, Bits)",
        "JSD (bits, 0 to 1)",
        "jsd_heatmap.png",
        OUTPUT_DIR,
    )
    generate_heatmap(
        composite_df,
        "Standardized Distributional Difference Score (0-5 hrs, Equal-Weighted Composite)",
        "Composite Score (0=Best, 1=Worst)",
        "composite_heatmap.png",
        OUTPUT_DIR,
        cmap="magma_r",
    )

    # Step 7: Generate Markdown Report
    print("\n--- STEP 7: COMPILING SCIENTIFIC MARKDOWN REPORT ---")
    report_path = os.path.join(OUTPUT_DIR, "kinematic_distribution_similarity_report_0to5hrs.md")
    write_markdown_report(
        report_path,
        raw_wass_df,
        raw_ks_df,
        raw_jsd_df,
        std_wass_df,
        std_ks_df,
        std_jsd_df,
        composite_df,
        category_df,
        rankings_dict,
        comp_summary_df,
        feature_stats_df,
        metadata_summary,
    )
    print(f" -> Comprehensive report saved: {report_path}")

    print("\n=========================================================================")
    print("ANALYSIS COMPLETE (0 TO 5 HOURS)")
    print(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=========================================================================\n")


# =============================================================================
# 7. SCIENTIFIC MARKDOWN REPORT GENERATOR
# =============================================================================
def write_markdown_report(
    filepath,
    raw_wass,
    raw_ks,
    raw_jsd,
    std_wass,
    std_ks,
    std_jsd,
    composite,
    category_df,
    rankings,
    comp_summary_df,
    feature_stats_df,
    metadata_summary,
):
    best_comp_name = comp_summary_df.iloc[0]["Comparison"]
    best_comp_score = comp_summary_df.iloc[0]["Overall_Composite_Score"]
    worst_comp_name = comp_summary_df.iloc[-1]["Comparison"]
    worst_comp_score = comp_summary_df.iloc[-1]["Overall_Composite_Score"]

    overall_cat_means = category_df.mean(axis=1).sort_values(ascending=True)
    best_cat_name = overall_cat_means.index[0]
    best_cat_score = overall_cat_means.iloc[0]

    mean_feat_composite = composite.mean(axis=1).sort_values(ascending=False)
    top_worst_feats = mean_feat_composite.head(5)

    dataset_rows = ""
    for name, meta in metadata_summary.items():
        dataset_rows += f"| {name} | {meta['sampled_rows']:,} | {len(meta['unique_tracks']):,} | {meta['obs_per_track']:.1f} |\n"

    raw_diag_text = ""
    for comp_name in COMPARISON_NAMES:
        c_stats = feature_stats_df[feature_stats_df["Comparison"] == comp_name]
        um_feats = c_stats[c_stats["Units"] == "µm"]
        max_w_um = um_feats.sort_values("Raw_Wasserstein", ascending=False).iloc[0]
        vel_feats = c_stats[c_stats["Units"] == "µm/s"]
        max_w_vel = vel_feats.sort_values("Raw_Wasserstein", ascending=False).iloc[0]

        raw_diag_text += (
            f"### {comp_name}\n"
            f"- **Largest Spatial Discrepancy (Wasserstein)**: `{max_w_um['Feature']}` with **{max_w_um['Raw_Wasserstein']:.3f} µm** "
            f"(Exp Mean: {max_w_um['Exp_Mean']:.3f} µm vs Sim Mean: {max_w_um['Sim_Mean']:.3f} µm).\n"
            f"- **Largest Velocity Discrepancy (Wasserstein)**: `{max_w_vel['Feature']}` with **{max_w_vel['Raw_Wasserstein']:.3f} µm/s** "
            f"(Exp Mean: {max_w_vel['Exp_Mean']:.3f} µm/s vs Sim Mean: {max_w_vel['Sim_Mean']:.3f} µm/s).\n\n"
        )

    rankings_text = ""
    for comp_name in COMPARISON_NAMES:
        rankings_text += f"\n### {comp_name}\n"
        rankings_text += rankings[comp_name][
            [
                "Rank",
                "Feature",
                "Units",
                "Composite_Score",
                "Raw_Wasserstein",
                "Std_Wasserstein",
                "Std_KS",
                "Std_JSD",
            ]
        ].to_markdown(index=False)
        rankings_text += "\n"

    report_content = f"""# Quantitative Kinematic Distribution Similarity Report (0 to 5 Hours)

## 1. Executive Summary
This report assesses the mathematical and physical similarity between regime-matched experimental live-cell tracking datasets and GPU simulation trajectories restricted strictly to the **first 5 hours of tracking** (`FRAME <= {FRAME_MAX_CUTOFF}`).

### Quality-Control Filtering Applied:
- **Cancer Cell Tracks**: Retains entire trajectories only if total tracking duration $\\ge {CANCER_MIN_SECONDS:.0f}$ s (15 minutes).
- **Immune / T-Cell Tracks**: Retains entire trajectories only if total tracking duration $\\ge {IMMUNE_MIN_SECONDS:.0f}$ s (1 hour).
- **Time Window**: Restricts downstream statistical evaluations to $0 \\le \\text{{FRAME}} \\le {FRAME_MAX_CUTOFF}$.

### Key Findings (0 - 5 Hours):
- **Best Overall Agreement**: **{best_comp_name}** (Composite Index = **{best_comp_score:.3f}**).
- **Worst Overall Agreement**: **{worst_comp_name}** (Composite Index = **{worst_comp_score:.3f}**).
- **Most Accurately Reproduced Category**: **{best_cat_name}** (Mean Category Composite = **{best_cat_score:.3f}**).
- **Top Calibration Targets**: `{top_worst_feats.index[0]}` (Mean Composite = **{top_worst_feats.iloc[0]:.3f}**) and `{top_worst_feats.index[1]}` (Mean Composite = **{top_worst_feats.iloc[1]:.3f}**).

---

## 2. Dataset Information and Observational Metadata (0-5 hrs)
| Dataset Description | Analyzed Observations | Unique Retained Tracks | Obs / Track |
| :--- | :--- | :--- | :--- |
{dataset_rows}

---

## 3. Overall Comparison Summary (0 - 5 Hours)
{comp_summary_df.to_markdown(index=False)}

---

## 4. Feature Category Similarity Matrix
{category_df.to_markdown()}

---

## 5. Primary Physical Realism Diagnostics (Raw Physical Units)
{raw_diag_text}

---

## 6. Feature Rankings by Regime Comparison (Best Match to Worst Match)
{rankings_text}

---

## 7. Raw Physical Unit Metric Tables

### Raw Wasserstein Distance ($W_1$ in physical units)
{raw_wass.to_markdown()}

### Raw Kolmogorov-Smirnov Statistic ($D_{{\\text{{KS}}}}$)
{raw_ks.to_markdown()}

### Raw Jensen-Shannon Divergence ($\\text{{JSD}}$ in bits)
{raw_jsd.to_markdown()}

---

## 8. Experimental Reference Standardized Z-Score Tables

### Standardized Wasserstein Distance ($W_1$ in $\\sigma_{{\\text{{exp}}}}$ units)
{std_wass.to_markdown()}

### Standardized Kolmogorov-Smirnov Statistic ($D_{{\\text{{KS}}}}$)
{std_ks.to_markdown()}

### Standardized Jensen-Shannon Divergence ($\\text{{JSD}}$ in bits)
{std_jsd.to_markdown()}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)


if __name__ == "__main__":
    main()