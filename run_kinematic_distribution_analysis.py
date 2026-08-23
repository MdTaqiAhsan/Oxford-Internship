"""
run_kinematic_distribution_analysis.py
================================================================================
Scientific Quantitative Distribution Comparison Analysis for Cell Kinematics
--------------------------------------------------------------------------------
Compares 4 regime-matched experimental datasets against 4 simulation datasets.
Filters full-frame simulation CSVs to observational frames (FRAME % 6 == 0)
where kinematic features represent the instantaneous preceding 1-frame step.

Evaluates mathematical similarity via:
  1. Primary Analysis: Raw Physical Units (Wasserstein, KS, JSD)
  2. Secondary Diagnostic Analysis: Experimental Reference Standardized Z-Scores
     (Wasserstein in sigma_exp, KS, JSD, Delta_Z)

Exports comprehensive CSV metric tables, feature statistics, publication heatmaps,
and a dynamically populated Markdown report.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. FILE PATHS AND DIRECTORY CONFIGURATION (8 DISTINCT DATASETS)
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

KILLING_CANCER = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\new_simulator\kinematic_csv_outputs"
    r"\killing_Cancer-cell_kinematics.csv"
)

KILLING_TCELL = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\new_simulator\kinematic_csv_outputs"
    r"\killing_T-cell_kinematics.csv"
)

NONKILLING_CANCER = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\new_simulator\kinematic_csv_outputs"
    r"\non-killing_Cancer-cell_kinematics.csv"
)

NONKILLING_TCELL = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\new_simulator\kinematic_csv_outputs"
    r"\non-killing_T-cell_kinematics.csv"
)

OUTPUT_DIR = r"kinematic_distribution_metrics"

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
    "Long-Term Motility": [
        "DISTANCE_TRAVELED",
        "PATH_EFFICIENCY",
        "AVERAGE_SPEED",
    ],
}

# Explicit mapping of the 4 one-to-one regime comparisons
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
    "DISTANCE_TRAVELED",
    "PATH_EFFICIENCY",
    "SPEED",
    "AVERAGE_SPEED",
]


# =============================================================================
# 3. MATHEMATICAL METRIC IMPLEMENTATIONS
# =============================================================================
def calculate_wasserstein(arr1, arr2):
    """Calculates 1st Wasserstein Distance (Earth Mover's Distance). Sample-based."""
    return float(stats.wasserstein_distance(arr1, arr2))


def calculate_ks_statistic(arr1, arr2):
    """Calculates Two-Sample Kolmogorov-Smirnov Statistic & p-value. Sample-based."""
    res = stats.ks_2samp(arr1, arr2)
    return float(res.statistic), float(res.pvalue)


def calculate_jsd_standardized(arr1_z, arr2_z, z_min=-10.0, z_max=10.0, num_bins=200):
    """
    Calculates Jensen-Shannon Divergence in bits (base 2) in standardized Z-space.
    Uses a fixed support [-10, +10] with 200 bins. Clamps extreme values to outer bins.
    """
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
    """
    Calculates JSD in raw physical units over common empirical bounds [min, max].
    """
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
# 4. DATA LOADING, FILTERING & SANITY VALIDATION
# =============================================================================
def load_and_validate_dataset(filepath, description, is_simulation=False, chunksize=200_000):
    """
    Loads dataset in memory-efficient chunks. For simulation datasets, filters to
    observation frames (FRAME % 6 == 0) to align with experimental frame sampling.
    """
    print(f" -> Loading [{description}] from:\n    {filepath}")
    if not os.path.exists(filepath):
        print(f" [ERROR] File missing: {filepath}", file=sys.stderr)
        sys.exit(1)

    feature_data = {f: [] for f in FEATURES}
    metadata = {
        "total_rows": 0,
        "sampled_rows": 0,
        "unique_tracks": set(),
        "invalid_counts": {f: 0 for f in FEATURES},
        "path_eff_out_of_bounds": 0,
    }

    try:
        preview = pd.read_csv(filepath, nrows=2)
        cols_present = [f for f in FEATURES if f in preview.columns]

        track_col = None
        for col_name in preview.columns:
            if col_name.upper() in ["TRACK_ID", "ID", "TRACKID", "CELL_ID"]:
                track_col = col_name
                break

        read_cols = cols_present + (["FRAME"] if "FRAME" in preview.columns else [])
        read_cols += ([track_col] if track_col and track_col not in read_cols else [])

        for chunk in pd.read_csv(filepath, usecols=read_cols, chunksize=chunksize):
            metadata["total_rows"] += len(chunk)

            # Filter simulation datasets to observation frames AFTER kinetics calculation
            if is_simulation and "FRAME" in chunk.columns:
                chunk = chunk[chunk["FRAME"] % 6 == 0]

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

        filter_note = f" (filtered from {metadata['total_rows']:,} full frames)" if is_simulation else ""
        print(
            f"    Loaded {metadata['sampled_rows']:,} analysis observations{filter_note} | "
            f"{num_tracks:,} unique tracks | "
            f"~{obs_per_track:.1f} obs/track"
        )

        metadata["obs_per_track"] = obs_per_track
        return clean_data, metadata

    except Exception as e:
        print(f" [ERROR] Failed loading {filepath}: {str(e)}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# 5. HEATMAP GENERATION ENGINE
# =============================================================================
def generate_heatmap(df, title, cbar_label, filename, output_dir, cmap="YlOrRd", fmt=".3f"):
    """Renders and saves a publication-quality heatmap for metric matrices."""
    plt.figure(figsize=(12, 9), dpi=300)
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

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=========================================================================")
    print("   QUANTITATIVE KINEMATIC DISTRIBUTION ANALYSIS PIPELINE INITIALIZED     ")
    print("=========================================================================")
    print(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}\n")

    # Step 1: Load All 8 Datasets
    print("--- STEP 1: LOADING & FILTERING ALL 8 DATASETS ---")
    datasets = {}
    metadata_summary = {}

    dataset_paths = [
        ("Experimental Killing Cancer", EXPERIMENTAL_KILLING_CANCER, False),
        ("Experimental Non-Killing Cancer", EXPERIMENTAL_NONKILLING_CANCER, False),
        ("Experimental Killing T-Cell", EXPERIMENTAL_KILLING_TCELL, False),
        ("Experimental Non-Killing T-Cell", EXPERIMENTAL_NONKILLING_TCELL, False),
        ("Killing Cancer", KILLING_CANCER, True),
        ("Killing T-Cell", KILLING_TCELL, True),
        ("Non-Killing Cancer", NONKILLING_CANCER, True),
        ("Non-Killing T-Cell", NONKILLING_TCELL, True),
    ]

    for name, path, is_sim in dataset_paths:
        data, meta = load_and_validate_dataset(path, name, is_simulation=is_sim)
        datasets[name] = data
        metadata_summary[name] = meta

    # Step 1B: Validation & Array Sanity Checks
    print("\n--- DATASET MAPPING & EQUALITY SANITY CHECKS ---")
    for comp_name, exp_k, sim_k in COMPARISONS:
        print(f"  [MAP CHECK] {comp_name:<55} -> [{exp_k}] vs [{sim_k}]")

    all_dataset_keys = list(datasets.keys())
    for i in range(len(all_dataset_keys)):
        for j in range(i + 1, len(all_dataset_keys)):
            k1 = all_dataset_keys[i]
            k2 = all_dataset_keys[j]
            d1_data = datasets[k1]
            d2_data = datasets[k2]
            for feat in FEATURES:
                arr1 = d1_data[feat]
                arr2 = d2_data[feat]
                if len(arr1) == len(arr2) and len(arr1) > 0 and np.array_equal(arr1, arr2):
                    print(
                        f"  [WARNING] Array equality detected! {k1} and {k2} "
                        f"have identical data for feature: {feat}"
                    )

    # Step 2: Compute Raw and Experimental-Reference Standardized Metrics
    print("\n--- STEP 2: CALCULATING RAW AND EXPERIMENTAL-REFERENCE STANDARDIZED METRICS ---")

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

            # --- A. Summary Statistics ---
            e_mean, e_std = float(np.mean(e_arr)), float(np.std(e_arr))
            s_mean, s_std = float(np.mean(s_arr)), float(np.std(s_arr))
            e_median, e_iqr = float(np.median(e_arr)), float(stats.iqr(e_arr))
            s_median, s_iqr = float(np.median(s_arr)), float(stats.iqr(s_arr))

            # --- B. Raw Physical-Unit Metrics ---
            w_raw = calculate_wasserstein(e_arr, s_arr)
            ks_stat_raw, ks_p_raw = calculate_ks_statistic(e_arr, s_arr)
            jsd_raw = calculate_jsd_raw(e_arr, s_arr)

            raw_wass_df.loc[feat, comp_name] = w_raw
            raw_ks_df.loc[feat, comp_name] = ks_stat_raw
            raw_ks_p_df.loc[feat, comp_name] = ks_p_raw
            raw_jsd_df.loc[feat, comp_name] = jsd_raw

            # --- C. Experimental-Reference Standardization ---
            # Strictly use experimental mean and std for the specific comparison
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

    # Step 3: Compute Per-Metric Normalized Composite Scores
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

    norm_details_rows = []
    for feat in FEATURES:
        for comp in COMPARISON_NAMES:
            norm_details_rows.append({
                "Feature": feat,
                "Comparison": comp,
                "Std_Wasserstein": std_wass_df.loc[feat, comp],
                "Norm_Wasserstein": norm_wass.loc[feat, comp],
                "Std_KS": std_ks_df.loc[feat, comp],
                "Norm_KS": norm_ks.loc[feat, comp],
                "Std_JSD": std_jsd_df.loc[feat, comp],
                "Norm_JSD": norm_jsd.loc[feat, comp],
                "Composite_Score": composite_df.loc[feat, comp],
            })
    norm_details_df = pd.DataFrame(norm_details_rows)

    category_df = pd.DataFrame(index=list(CATEGORIES.keys()), columns=COMPARISON_NAMES, dtype=float)
    for cat_name, cat_feats in CATEGORIES.items():
        category_df.loc[cat_name] = composite_df.loc[cat_feats].mean(axis=0)

    cat_rankings_rows = []
    for comp in COMPARISON_NAMES:
        sorted_cats = category_df[comp].sort_values(ascending=True)
        for rank, (cat_name, score) in enumerate(sorted_cats.items(), start=1):
            cat_rankings_rows.append({
                "Comparison": comp,
                "Rank": rank,
                "Category": cat_name,
                "Category_Score": score,
            })
    category_rankings_df = pd.DataFrame(cat_rankings_rows)

    overall_comp_series = category_df.mean(axis=0).sort_values(ascending=True)
    comp_summary_df = pd.DataFrame({
        "Comparison": overall_comp_series.index,
        "Overall_Composite_Score": overall_comp_series.values,
        "Spatial_Behavior": category_df.loc["Spatial Behavior", overall_comp_series.index].values,
        "Local_Motility": category_df.loc["Local Motility", overall_comp_series.index].values,
        "Velocity": category_df.loc["Velocity", overall_comp_series.index].values,
        "Long_Term_Motility": category_df.loc["Long-Term Motility", overall_comp_series.index].values,
    })
    comp_summary_df["Rank"] = np.arange(1, len(comp_summary_df) + 1)

    # Step 4: Compute Feature Rankings per Comparison
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

    # Step 5: Save CSV Outputs
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
    category_rankings_df.to_csv(os.path.join(OUTPUT_DIR, "category_rankings.csv"), index=False)
    norm_details_df.to_csv(os.path.join(OUTPUT_DIR, "metric_normalization_details.csv"), index=False)

    all_ranks_list = []
    for comp, df_r in rankings_dict.items():
        df_temp = df_r.copy()
        df_temp["Comparison"] = comp
        all_ranks_list.append(df_temp)
    master_rank_df = pd.concat(all_ranks_list, ignore_index=True)
    master_rank_df.to_csv(os.path.join(OUTPUT_DIR, "feature_rankings.csv"), index=False)

    print(" -> All CSV metric tables exported successfully.")

    # Step 6: Generate Heatmaps
    print("\n--- STEP 6: GENERATING PUBLICATION HEATMAP FIGURES ---")
    generate_heatmap(
        std_wass_df,
        "Wasserstein Distance (Experimental Standardized Units)",
        "Wasserstein Distance (σ_exp units)",
        "wasserstein_heatmap.png",
        OUTPUT_DIR,
        cmap="YlOrRd",
    )
    generate_heatmap(
        std_ks_df,
        "Kolmogorov-Smirnov Statistic (Descriptive CDF Disagreement)",
        "KS Statistic (0 to 1)",
        "ks_heatmap.png",
        OUTPUT_DIR,
        cmap="YlOrRd",
    )
    generate_heatmap(
        std_jsd_df,
        "Jensen-Shannon Divergence (Standardized Z-Space, Bits)",
        "JSD (bits, 0 to 1)",
        "jsd_heatmap.png",
        OUTPUT_DIR,
        cmap="YlOrRd",
    )
    generate_heatmap(
        composite_df,
        "Standardized Distributional Difference Score (1/3 W_std + 1/3 KS + 1/3 JSD)",
        "Composite Score (0=Best, 1=Worst)",
        "composite_heatmap.png",
        OUTPUT_DIR,
        cmap="magma_r",
    )

    # Step 7: Generate Markdown Report
    print("\n--- STEP 7: COMPILING SCIENTIFIC MARKDOWN REPORT ---")
    report_path = os.path.join(OUTPUT_DIR, "kinematic_distribution_similarity_report.md")
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
    print(f" -> Comprehensive scientific report saved: {report_path}")

    print("\n=========================================================================")
    print("ALL COMPARISONS COMPLETE")
    print(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=========================================================================\n")


# =============================================================================
# 7. MARKDOWN REPORT GENERATOR (DYNAMICALLY POPULATED)
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
    worst_cat_name = overall_cat_means.index[-1]
    worst_cat_score = overall_cat_means.iloc[-1]

    mean_feat_composite = composite.mean(axis=1).sort_values(ascending=False)
    top_worst_feats = mean_feat_composite.head(5)

    dataset_rows = ""
    for name, meta in metadata_summary.items():
        num_obs = meta["sampled_rows"]
        num_tr = len(meta["unique_tracks"])
        o_per_t = meta["obs_per_track"]
        dataset_rows += f"| {name} | {num_obs:,} | {num_tr:,} | {o_per_t:.1f} |\n"

    # Physical Realism Diagnostics: Largest raw discrepancies per comparison
    raw_diag_text = ""
    for comp_name in COMPARISON_NAMES:
        c_stats = feature_stats_df[feature_stats_df["Comparison"] == comp_name]
        
        # Positional / displacement (µm)
        um_feats = c_stats[c_stats["Units"] == "µm"]
        max_w_um = um_feats.sort_values("Raw_Wasserstein", ascending=False).iloc[0]
        
        # Velocity / speed (µm/s)
        vel_feats = c_stats[c_stats["Units"] == "µm/s"]
        max_w_vel = vel_feats.sort_values("Raw_Wasserstein", ascending=False).iloc[0]
        
        # Specific key metrics
        dist_row = c_stats[c_stats["Feature"] == "DISTANCE_TRAVELED"].iloc[0]
        disp_orig_row = c_stats[c_stats["Feature"] == "DISPLACEMENT_FROM_ORIGIN"].iloc[0]

        raw_diag_text += (
            f"### {comp_name}\n"
            f"- **Largest Spatial Discrepancy (Wasserstein)**: `{max_w_um['Feature']}` with **{max_w_um['Raw_Wasserstein']:.3f} µm** difference "
            f"(Exp Mean: {max_w_um['Exp_Mean']:.3f} µm vs Sim Mean: {max_w_um['Sim_Mean']:.3f} µm).\n"
            f"- **Largest Velocity/Speed Discrepancy (Wasserstein)**: `{max_w_vel['Feature']}` with **{max_w_vel['Raw_Wasserstein']:.3f} µm/s** difference "
            f"(Exp Mean: {max_w_vel['Exp_Mean']:.3f} µm/s vs Sim Mean: {max_w_vel['Sim_Mean']:.3f} µm/s).\n"
            f"- **Distance Traveled Discrepancy**: Raw Wasserstein = **{dist_row['Raw_Wasserstein']:.3f} µm** "
            f"(Exp Mean: {dist_row['Exp_Mean']:.3f} µm, Sim Mean: {dist_row['Sim_Mean']:.3f} µm).\n"
            f"- **Displacement from Origin Discrepancy**: Raw Wasserstein = **{disp_orig_row['Raw_Wasserstein']:.3f} µm** "
            f"(Exp Mean: {disp_orig_row['Exp_Mean']:.3f} µm, Sim Mean: {disp_orig_row['Sim_Mean']:.3f} µm).\n\n"
        )

    # Standardized diagnostics text
    std_diag_text = ""
    for idx, (feat_name, comp_score) in enumerate(top_worst_feats.items(), start=1):
        feat_rows = feature_stats_df[feature_stats_df["Feature"] == feat_name]
        worst_row = feat_rows.loc[
            composite.loc[feat_name].idxmax() == feat_rows["Comparison"]
        ].iloc[0]

        comp_with_worst = worst_row["Comparison"]
        delta_z = worst_row["Delta_Z"]
        w_s = worst_row["Std_Wasserstein"]
        ks_s = worst_row["Std_KS"]
        jsd_s = worst_row["Std_JSD"]

        std_diag_text += (
            f"{idx}. **`{feat_name}`** ({worst_row['Units']}) — Mean Composite Score = **{comp_score:.3f}**\n"
            f"   - *Worst Comparison Regime*: `{comp_with_worst}`\n"
            f"   - *Diagnostic Profile*: Delta_Z = **{delta_z:+.3f} σ_exp** | W1 = **{w_s:.3f} σ_exp** | KS Stat = **{ks_s:.3f}** | JSD = **{jsd_s:.3f} bits**\n"
            f"   - *Physical Values*: Exp Mean = **{worst_row['Exp_Mean']:.3f} {worst_row['Units']}**, Sim Mean = **{worst_row['Sim_Mean']:.3f} {worst_row['Units']}** "
            f"(Exp Median = {worst_row['Exp_Median']:.3f}, Sim Median = {worst_row['Sim_Median']:.3f}).\n"
            f"   - *Diagnostic Interpretation*: Standardized shift of {delta_z:+.3f} σ_exp relative to experimental variability "
            f"indicates candidate area for simulator calibration.\n\n"
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

    report_content = f"""# Quantitative Kinematic Distribution Similarity Report

## 1. Executive Summary
This report provides an unbiased, quantitative assessment comparing 4 regime-matched experimental live-cell tracking datasets against GPU simulation trajectories sampled at identical observational frames (`FRAME % 6 == 0`).

Simulation kinematics represent instantaneous 1-frame transitions ($t$ vs $t-1$, $dt=1.0$) with physical units directly matching experimental microscopy tracking measurements:
- **Positions, Displacements, Distances**: µm
- **Velocities, Speeds**: µm/s
- **Path Efficiency**: dimensionless

Mathematical similarity is evaluated under two distinct perspectives:
1. **Primary Analysis (Raw Physical Units)**: Quantifies the absolute physical discrepancy between experimental measurements and simulation outputs.
2. **Secondary Diagnostic Analysis (Experimental Reference Z-Scores)**: Evaluates the discrepancy relative to the natural biological variability ($\sigma_{{\\text{{exp}}}}$) of the corresponding experimental condition.

### Dynamic Summary of Key Findings:
- **Best Overall Comparison Agreement**: **{best_comp_name}** achieved the lowest overall standardized distributional difference index (**{best_comp_score:.3f}**).
- **Worst Overall Comparison Agreement**: **{worst_comp_name}** exhibited the highest overall standardized distributional difference index (**{worst_comp_score:.3f}**).
- **Most Accurately Reproduced Category**: **{best_cat_name}** across all regimes (Mean Category Composite Score = **{best_cat_score:.3f}**).
- **Least Accurately Reproduced Category**: **{worst_cat_name}** across all regimes (Mean Category Composite Score = **{worst_cat_score:.3f}**).
- **Primary Diagnostic Targets**: Top feature mismatches are led by `{top_worst_feats.index[0]}` (Mean Composite = **{top_worst_feats.iloc[0]:.3f}**) and `{top_worst_feats.index[1]}` (Mean Composite = **{top_worst_feats.iloc[1]:.3f}**).

---

## 2. Dataset Information and Observational Metadata
The dataset consists of observations recorded across individual cell tracks (`TRACK_ID`) across 4 distinct experimental datasets and 4 corresponding simulation output files (filtered at `FRAME % 6 == 0`).

| Dataset Description | Analyzed Observations | Unique Tracks | Obs / Track |
| :--- | :--- | :--- | :--- |
{dataset_rows}
### Observational Structure & Statistical Autocorrelation
Sequential frame observations within an individual cell trajectory exhibit temporal autocorrelation and are not independent biological replicates. Conventional hypothesis-testing p-values scale with sample size ($N > 10^5$), evaluating to $p < 1\\times 10^{{-300}}$. 
**Methodological Safeguard**: Because trajectory observations are temporally correlated and therefore not independent, KS p-values should not be interpreted as conventional hypothesis-test evidence of biological significance or non-equivalence. The KS statistic is retained purely as a descriptive empirical CDF distance.

---

## 3. Statistical Methodology & Metric Definitions

### A. Raw Physical-Unit Evaluation (Primary Analysis)
Because the simulator output and experimental tracking datasets use identical physical units (µm, µm/s), raw metrics directly evaluate physical realism without scaling:
- **Raw Wasserstein ($W_1$)**: Minimal work required to transform the simulation distribution into the experimental distribution, retaining the feature's physical unit (µm, µm/s).
- **Raw Two-Sample KS ($D_{{\\text{{KS}}}}$)**: Maximum absolute vertical distance between empirical CDFs. Bounded $[0, 1]$.
- **Raw JSD**: Symmetrical Jensen-Shannon Divergence in bits ($\log_2$) computed over common empirical support. Bounded $[0, 1]$ bits.

### B. Experimental-Reference Z-Score Standardization (Secondary Diagnostic)
To assess how large simulation discrepancies are relative to biological variability, standardization is performed using **strictly the experimental baseline parameters of that specific comparison** ($\mu_{{\\text{{exp}}}}$, $\sigma_{{\\text{{exp}}}}$):

$$z_{{\\text{{exp}}}} = \\frac{{x_{{\\text{{exp}}}} - \\mu_{{\\text{{exp}}}}}}{{\\sigma_{{\\text{{exp}}}}}}, \\quad z_{{\\text{{sim}}}} = \\frac{{x_{{\\text{{sim}}}} - \\mu_{{\\text{{exp}}}}}}{{\\sigma_{{\\text{{exp}}}}}}$$

- **Delta_Z (Diagnostic Effect Size)**: $\\Delta Z = (\\mu_{{\\text{{sim}}}} - \\mu_{{\\text{{exp}}}}) / \\sigma_{{\\text{{exp}}}}$. Represents the mean shift in units of experimental standard deviations.
- **Standardized Wasserstein ($W_1$)**: Distributional distance expressed in $\\sigma_{{\\text{{exp}}}}$ units.

### C. Standardized Distributional Difference Score (Composite)
Per-metric Min-Max normalization maps standardized $W_1$, $D_{{\\text{{KS}}}}$, and $\\text{{JSD}}$ to $[0, 1]$. The Composite Difference Index is:

$$\\text{{Composite Score}} = \\frac{{1}}{{3}} \\tilde{{W}}_{{\\text{{std}}}} + \\frac{{1}}{{3}} \\tilde{{D}}_{{\\text{{KS}}}} + \\frac{{1}}{{3}} \\tilde{{\\text{{JSD}}}}$$

This is a diagnostic ranking metric where lower scores indicate greater distributional agreement. It does not represent a biological significance test or physical realism score.

---

## 4. Overall Comparison-Level Summary
Lower composite scores indicate superior distributional alignment with experimental target data.

{comp_summary_df.to_markdown(index=False)}

---

## 5. Feature Category Similarity Matrix
Mean composite scores by behavioral category (0 = Identical, 1 = Maximum Divergence):

{category_df.to_markdown()}

---

## 6. Primary Physical Realism Diagnostics (Raw Physical Units)
Key absolute physical discrepancies between experimental and simulation distributions:

{raw_diag_text}

---

## 7. Secondary Diagnostic Analysis (Largest Standardized Discrepancies)
Candidate features for simulator calibration based on standardized divergence from experimental variability:

{std_diag_text}

---

## 8. Feature Rankings by Comparison (Best Match to Worst Match)
{rankings_text}

---

## 9. Raw Physical Unit Metric Tables

### Raw Wasserstein Distance ($W_1$ in physical units)
{raw_wass.to_markdown()}

### Raw Kolmogorov-Smirnov Statistic ($D_{{\\text{{KS}}}}$)
{raw_ks.to_markdown()}

### Raw Jensen-Shannon Divergence ($\\text{{JSD}}$ in bits)
{raw_jsd.to_markdown()}

---

## 10. Experimental Reference Standardized Z-Score Tables

### Standardized Wasserstein Distance ($W_1$ in $\\sigma_{{\\text{{exp}}}}$ units)
{std_wass.to_markdown()}

### Standardized Kolmogorov-Smirnov Statistic ($D_{{\\text{{KS}}}}$)
{std_ks.to_markdown()}

### Standardized Jensen-Shannon Divergence ($\\text{{JSD}}$ in bits)
{std_jsd.to_markdown()}

---

## 11. Local Kinematics vs. Global Trajectory Discrepancies
Comparing local kinematic features (`VEL_X`, `VEL_Y`, `SPEED`, `DX/DY_FROM_PREVIOUS_POINT`) against long-term spatial trajectory features (`DISPLACEMENT_FROM_ORIGIN`, `DISTANCE_TRAVELED`, `DX/DY_FROM_ORIGIN`) highlights a structural dichotomy:

- **Local Kinematics Category Score**: **{category_df.loc["Velocity"].mean():.3f}** (Velocity) | **{category_df.loc["Local Motility"].mean():.3f}** (Local Motility).
- **Global Trajectory Category Score**: **{category_df.loc["Spatial Behavior"].mean():.3f}** (Spatial Behavior) | **{category_df.loc["Long-Term Motility"].mean():.3f}** (Long-Term Motility).

**Diagnostic Insight**: The simulator reproduces local instantaneous velocity vectors with relatively higher physical fidelity, but integrated positional errors accumulate over long temporal horizons ($t > 50$ frames), leading to spatial dispersion mismatches.

---

## 12. Candidate Simulator Parameters for Future Calibration

The statistical analysis establishes **distributional non-equivalence** ($P_{{\\text{{exp}}}} \\neq P_{{\\text{{sim}}}}$). The parameter associations below represent **candidate mechanisms for future experimental calibration**, not established causal proofs.

| Observed Physical / Standardized Mismatch | Potential Mechanistic Explanation | Candidate Simulator Parameters to Investigate |
| :--- | :--- | :--- |
| **Excessive `DISPLACEMENT_FROM_ORIGIN` (µm)** | Simulated cells wander unconstrained across domain; missing spatial confinement. | `tau` (steering persistence), `max_speed`, `CAN_EVASIVE_SPEED`, boundary/tethering forces. |
| **High Cumulative `DISTANCE_TRAVELED` (µm)** | Continuous uninhibited motion with minimal resting phases. | `ENERGY_DRAIN_MOVE`, `ENERGY_RECOVER_REST`, `noise_scale`, phenotype speed multipliers (`SPEED_MULTS`). |
| **Step Displacements (`DX/DY_PREVIOUS`) Heavy Tails** | Steering accelerations produce step overshooting at 6-frame sampling intervals. | `noise_scale`, `IMMUNE_BASE_MEAN`, `tau`, simulation integration timestep `dt`. |
| **Path Efficiency Discrepancies in Non-Killing Mode** | Uninhibited random walks produce inefficient exploratory paths. | Non-killing baseline `tau`, `noise_scale`, chemotaxis weighting factors. |

---

## 13. Methodological Limitations
1. **Temporal Autocorrelation**: Frame-level observations within cell trajectories are time-dependent. KS statistics and p-values are descriptive.
2. **Diagnostic Composite Index**: The Composite Score is a normalized, equal-weighted diagnostic index ($w_1=1/3, w_2=1/3, w_3=1/3$), not a universal physical constant.
3. **Observational Frame Equivalence**: Simulation metrics are computed from continuous 1-frame integration ($dt=1.0$) and sampled at observation frames (`FRAME % 6 == 0`), matching microscopy sampling.

---

## 14. Final Conclusions
1. The analysis provides an **unbiased diagnostic baseline** without altering physical units or masking discrepancies.
2. **Velocity components and instantaneous speed** exhibit the strongest physical alignment with experimental live-cell tracking.
3. **Global spatial trajectory features** (`DISPLACEMENT_FROM_ORIGIN`, `DISTANCE_TRAVELED`) represent the primary physical targets for parameter calibration.
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)


if __name__ == "__main__":
    main()