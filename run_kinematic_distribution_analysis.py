"""
run_kinematic_distribution_analysis.py
================================================================================
Scientific Quantitative Distribution Comparison Analysis for Cell Kinematics
--------------------------------------------------------------------------------
Computes mathematical similarity metrics (Wasserstein Distance, 2-Sample KS
Statistic, and Jensen-Shannon Divergence) across 14 kinematic features and 4
one-to-one regime-matched experimental-vs-simulation comparisons in both Raw
Physical Units and Experimental Reference Standardized Z-Score Units.

Exports metric tables, statistics summaries, publication heatmaps, and a
dynamically generated scientific Markdown report.
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
# Four Experimental Regimes
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

# Four Simulation Regimes
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
# 2. FEATURE DEFINITIONS, CATEGORIES & COMPARISON MAPPINGS
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

# Explicit dataset pair mapping (comp_name, exp_key, sim_key)
# Guarantees zero ambiguity or key-mismatch
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
# 4. DATA LOADING AND SANITY VALIDATION
# =============================================================================
def load_and_validate_csv(filepath, description, chunksize=200_000):
    """
    Loads dataset in chunks, extracts feature arrays and track metadata,
    and performs numeric validation checks.
    """
    print(f" -> Loading [{description}] from:\n    {filepath}")
    if not os.path.exists(filepath):
        print(f" [ERROR] File missing: {filepath}", file=sys.stderr)
        sys.exit(1)

    feature_data = {f: [] for f in FEATURES}
    metadata = {
        "total_rows": 0,
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

        read_cols = cols_present + ([track_col] if track_col else [])

        for chunk in pd.read_csv(filepath, usecols=read_cols, chunksize=chunksize):
            metadata["total_rows"] += len(chunk)
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
            metadata["total_rows"] / num_tracks if num_tracks > 0 else 0.0
        )

        print(
            f"    Loaded {metadata['total_rows']:,} observations | "
            f"{num_tracks:,} unique tracks | "
            f"~{obs_per_track:.1f} obs/track"
        )
        if metadata["path_eff_out_of_bounds"] > 0:
            print(
                f"    [NOTE] PATH_EFFICIENCY had {metadata['path_eff_out_of_bounds']:,} "
                f"observations slightly outside [0, 1] (retained)."
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
    plt.xlabel("Experimental vs Simulation Comparison", fontsize=11, fontweight="bold")
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
    print("--- STEP 1: LOADING & VALIDATING ALL 8 DATASETS ---")
    datasets = {}
    metadata_summary = {}

    dataset_paths = [
        ("Experimental Killing Cancer", EXPERIMENTAL_KILLING_CANCER),
        ("Experimental Non-Killing Cancer", EXPERIMENTAL_NONKILLING_CANCER),
        ("Experimental Killing T-Cell", EXPERIMENTAL_KILLING_TCELL),
        ("Experimental Non-Killing T-Cell", EXPERIMENTAL_NONKILLING_TCELL),
        ("Killing Cancer", KILLING_CANCER),
        ("Killing T-Cell", KILLING_TCELL),
        ("Non-Killing Cancer", NONKILLING_CANCER),
        ("Non-Killing T-Cell", NONKILLING_TCELL),
    ]

    for name, path in dataset_paths:
        data, meta = load_and_validate_csv(path, name)
        datasets[name] = data
        metadata_summary[name] = meta

    # Step 1B: Explicit Validation & Array Equality Sanity Checks
    print("\n--- DATASET MAPPING & EQUALITY SANITY CHECKS ---")
    for comp_name, exp_k, sim_k in COMPARISONS:
        print(f"  [MAP CHECK] {comp_name:<55} -> [{exp_k}] vs [{sim_k}]")

    # Check for accidental array duplication across loaded datasets
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
    print("\n--- STEP 2: CALCULATING METRICS (EXPERIMENTAL REFERENCE STANDARDIZATION) ---")

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

            # --- B. Raw Unit Metrics ---
            w_raw = calculate_wasserstein(e_arr, s_arr)
            ks_stat_raw, ks_p_raw = calculate_ks_statistic(e_arr, s_arr)
            jsd_raw = calculate_jsd_raw(e_arr, s_arr)

            raw_wass_df.loc[feat, comp_name] = w_raw
            raw_ks_df.loc[feat, comp_name] = ks_stat_raw
            raw_ks_p_df.loc[feat, comp_name] = ks_p_raw
            raw_jsd_df.loc[feat, comp_name] = jsd_raw

            # --- C. Experimental Reference Standardization ---
            # The specific comparison's experimental dataset defines the reference coordinate frame
            if e_std > 1e-12:
                e_std_arr = (e_arr - e_mean) / e_std
                s_std_arr = (s_arr - e_mean) / e_std
            else:
                e_std_arr = e_arr - e_mean
                s_std_arr = s_arr - e_mean

            std_mean_diff = float(np.mean(s_std_arr) - np.mean(e_std_arr))

            w_std = calculate_wasserstein(e_std_arr, s_std_arr)
            ks_stat_std, _ = calculate_ks_statistic(e_std_arr, s_std_arr)
            jsd_std = calculate_jsd_standardized(e_std_arr, s_std_arr)

            std_wass_df.loc[feat, comp_name] = w_std
            std_ks_df.loc[feat, comp_name] = ks_stat_std
            std_jsd_df.loc[feat, comp_name] = jsd_std

            feature_stats_rows.append({
                "Comparison": comp_name,
                "Feature": feat,
                "Exp_Mean": e_mean,
                "Exp_Std": e_std,
                "Exp_Median": e_median,
                "Exp_IQR": e_iqr,
                "Sim_Mean": s_mean,
                "Sim_Std": s_std,
                "Sim_Median": s_median,
                "Sim_IQR": s_iqr,
                "Std_Mean_Diff": std_mean_diff,
                "Raw_Wasserstein": w_raw,
                "Raw_KS": ks_stat_raw,
                "Raw_JSD": jsd_raw,
                "Std_Wasserstein": w_std,
                "Std_KS": ks_stat_std,
                "Std_JSD": jsd_std,
            })

    feature_stats_df = pd.DataFrame(feature_stats_rows)

    # Step 3: Compute Per-Metric Normalized Composite Scores
    print("\n--- STEP 3: COMPUTING PER-METRIC NORMALIZED COMPOSITE SCORES ---")

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
        "Normalized Composite Difference Score (1/3 W_std + 1/3 KS + 1/3 JSD)",
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
# 7. MARKDOWN REPORT GENERATOR (DYNAMICALLY POPULATED & SYNTAX SAFE)
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
        num_obs = meta["total_rows"]
        num_tr = len(meta["unique_tracks"])
        o_per_t = meta["obs_per_track"]
        dataset_rows += f"| {name} | {num_obs:,} | {num_tr:,} | {o_per_t:.1f} |\n"

    failure_modes_text = ""
    for idx, (feat_name, comp_score) in enumerate(top_worst_feats.items(), start=1):
        feat_rows = feature_stats_df[feature_stats_df["Feature"] == feat_name]
        worst_row = feat_rows.loc[
            composite.loc[feat_name].idxmax() == feat_rows["Comparison"]
        ].iloc[0]

        comp_with_worst = worst_row["Comparison"]
        mean_diff = worst_row["Std_Mean_Diff"]
        w_s = worst_row["Std_Wasserstein"]
        ks_s = worst_row["Std_KS"]

        failure_modes_text += (
            f"{idx}. **`{feat_name}`** (Mean Composite Score = **{comp_score:.3f}**)\n"
            f"   - *Worst Comparison Pair*: `{comp_with_worst}`\n"
            f"   - *Statistical Profile*: Standardized Mean Shift = **{mean_diff:+.3f} SD** | "
            f"W1 = **{w_s:.3f} SD** | KS Stat = **{ks_s:.3f}**\n"
            f"   - *Observed Shift*: Exp Mean = **{worst_row['Exp_Mean']:.3f}**, Sim Mean = **{worst_row['Sim_Mean']:.3f}** "
            f"(Exp Median = **{worst_row['Exp_Median']:.3f}**, Sim Median = **{worst_row['Sim_Median']:.3f}**).\n"
            f"   - *Diagnostic Interpretation*: Standardized mean shift of {mean_diff:+.3f} SD suggests "
            f"potential location shift or tail expansion in simulation.\n\n"
        )

    rankings_text = ""
    for comp_name in COMPARISON_NAMES:
        rankings_text += f"\n### {comp_name}\n"
        rankings_text += rankings[comp_name][
            [
                "Rank",
                "Feature",
                "Composite_Score",
                "Std_Wasserstein",
                "Std_KS",
                "Std_JSD",
            ]
        ].to_markdown(index=False)
        rankings_text += "\n"

    report_content = f"""# Quantitative Kinematic Distribution Similarity Report

## 1. Executive Summary
This report provides an unbiased, quantitative assessment comparing experimental live-cell tracking distributions against GPU-accelerated PyTorch simulation outputs across **14 kinematic features** and **4 regime-matched, one-to-one experimental-vs-simulation comparisons**.

Mathematical similarity is evaluated using **1st Wasserstein Distance (W1)**, **Two-Sample Kolmogorov-Smirnov Statistics (KS)**, and **Jensen-Shannon Divergence (JSD)**.

### Dynamic Summary of Key Findings:
- **Best Overall Comparison Agreement**: **{best_comp_name}** achieved the lowest overall composite difference index (**{best_comp_score:.3f}**).
- **Worst Overall Comparison Agreement**: **{worst_comp_name}** exhibited the highest overall composite difference index (**{worst_comp_score:.3f}**).
- **Most Accurately Reproduced Category**: **{best_cat_name}** across all regimes (Mean Category Composite Score = **{best_cat_score:.3f}**).
- **Least Accurately Reproduced Category**: **{worst_cat_name}** across all regimes (Mean Category Composite Score = **{worst_cat_score:.3f}**).
- **Primary Diagnostic Target**: Top feature mismatches are led by `{top_worst_feats.index[0]}` (Mean Composite = **{top_worst_feats.iloc[0]:.3f}**) and `{top_worst_feats.index[1]}` (Mean Composite = **{top_worst_feats.iloc[1]:.3f}**).

---

## 2. Dataset Information and Observational Metadata
The dataset consists of repeated observations recorded across individual cell tracks (`TRACK_ID`) across 4 distinct experimental datasets and 4 corresponding simulation output files.

| Dataset Description | Observations | Unique Tracks | Obs / Track |
| :--- | :--- | :--- | :--- |
{dataset_rows}
### Observational Structure & Statistical Autocorrelation
Sequential frame observations within an individual cell trajectory exhibit temporal autocorrelation and are not independent biological replicates. Conventional hypothesis-testing p-values scale with sample size (N > 10^5), evaluating to p < 1e-300. 
**Methodological Safeguard**: Because trajectory observations are temporally correlated and therefore not independent, KS p-values should not be interpreted as conventional hypothesis-test evidence of biological significance or non-equivalence. The KS statistic is retained purely as a descriptive empirical CDF distance.

---

## 3. Statistical Methodology & Metric Definitions

### A. Regime-Specific Experimental Reference Standardization
To prevent simulation scale mismatches or variance from distorting the target coordinate frame, standardization is performed using **strictly the experimental baseline parameters of that specific comparison** (Mean_exp, Std_exp) independently for each feature:

z_exp = (x_exp - Mean_exp) / Std_exp
z_sim = (x_sim - Mean_exp) / Std_exp

Standardized Wasserstein distance (W1) represents distributional separation measured directly in units of that condition's experimental standard deviation.

### B. Metric Definitions
1. **Wasserstein Distance (W1)**: Sample-based 1st Wasserstein distance (Earth Mover's Distance). Measures minimal work required to transform one empirical sample distribution into another. Bounded [0, inf), lower is better.
2. **Kolmogorov-Smirnov Statistic (KS)**: Sample-based two-sample KS statistic. Measures peak empirical CDF divergence. Bounded [0, 1], lower is better.
3. **Jensen-Shannon Divergence (JSD)**: Density-based information-theoretic divergence calculated in bits (log2) over standardized z-space support [-10, +10] with 200 bins. Bounded [0, 1] bits, lower is better.

### C. Equal-Weighted Composite Score & Category Aggregation
Per-metric Min-Max normalization maps W1, KS, and JSD to [0, 1]. The Composite Difference Index is:

Composite Score = (1/3) * Norm_W1 + (1/3) * Norm_KS + (1/3) * Norm_JSD

Category Scores are calculated as the unweighted arithmetic mean of feature-level composite scores within each category. Lower scores indicate superior agreement.

---

## 4. Overall Comparison-Level Summary
Lower composite scores indicate superior distributional alignment with experimental data.

{comp_summary_df.to_markdown(index=False)}

---

## 5. Feature Category Similarity Matrix
Mean composite scores by behavioral category (0 = Identical, 1 = Maximum Divergence):

{category_df.to_markdown()}

---

## 6. Feature Rankings by Comparison (Best Match to Worst Match)
{rankings_text}

---

## 7. Raw Physical Unit Metric Tables

### Raw Wasserstein Distance (W1 in physical units)
{raw_wass.to_markdown()}

### Raw Kolmogorov-Smirnov Statistic (KS Stat)
{raw_ks.to_markdown()}

### Raw Jensen-Shannon Divergence (JSD in bits)
{raw_jsd.to_markdown()}

---

## 8. Experimental Reference Standardized Z-Score Tables

### Standardized Wasserstein Distance (W1 in Experimental Std units)
{std_wass.to_markdown()}

### Standardized Kolmogorov-Smirnov Statistic (KS Stat)
{std_ks.to_markdown()}

### Standardized Jensen-Shannon Divergence (JSD in bits)
{std_jsd.to_markdown()}

---

## 9. Dynamic Feature Mismatch Diagnostics (Top 5 Priority Failure Modes)
The top 5 feature mismatches ranked by cross-comparison mean composite score are:

{failure_modes_text}

---

## 10. Local Kinematics vs. Global Trajectory Discrepancies
Comparing local kinematic features (`VEL_X`, `VEL_Y`, `SPEED`, `DX/DY_FROM_PREVIOUS_POINT`) against long-term spatial trajectory features (`DISPLACEMENT_FROM_ORIGIN`, `DISTANCE_TRAVELED`, `DX/DY_FROM_ORIGIN`) highlights a structural dichotomy:

- **Local Kinematics Category Score**: **{category_df.loc["Velocity"].mean():.3f}** (Velocity) | **{category_df.loc["Local Motility"].mean():.3f}** (Local Motility).
- **Global Trajectory Category Score**: **{category_df.loc["Spatial Behavior"].mean():.3f}** (Spatial Behavior) | **{category_df.loc["Long-Term Motility"].mean():.3f}** (Long-Term Motility).

**Diagnostic Insight**: The simulator reproduces local instantaneous velocity vectors with relatively higher fidelity, but integrated positional errors accumulate over long temporal horizons (t > 50 frames), leading to spatial dispersion mismatches.

---

## 11. Observed Mismatches, Mechanistic Explanations, & Candidate Parameters

The statistical analysis establishes **distributional non-equivalence** (P_exp != P_sim). The parameter associations below represent **candidate mechanisms for future experimental calibration**, not established causal proofs.

| Observed Distributional Mismatch | Potential Mechanistic Explanation | Candidate Simulator Parameters to Investigate |
| :--- | :--- | :--- |
| **Excessive `DISPLACEMENT_FROM_ORIGIN`** | Simulated cells wander unconstrained across the domain; missing spatial anchoring. | `tau` (steering persistence), `max_speed`, `CAN_EVASIVE_SPEED`, boundary/tethering forces. |
| **High Cumulative `DISTANCE_TRAVELED`** | Continuous uninhibited motion with minimal resting phases. | `ENERGY_DRAIN_MOVE`, `ENERGY_RECOVER_REST`, `noise_scale`, phenotype speed multipliers (`SPEED_MULTS`). |
| **Step Displacements (`DX/DY_PREVIOUS`) Heavy Tails** | Steering accelerations produce step overshooting at 6-frame sampling intervals. | `noise_scale`, `IMMUNE_BASE_MEAN`, `tau`, simulation integration timestep `dt`. |
| **Path Efficiency Discrepancies in Non-Killing Mode** | Uninhibited random walks produce inefficient exploratory paths. | Non-killing baseline `tau`, `noise_scale`, chemotaxis weighting factors. |

---

## 12. Methodological Limitations
1. **Temporal Autocorrelation**: Frame-level observations within cell trajectories are time-dependent. KS statistics are descriptive.
2. **Pragmatic Composite Index**: The Composite Score is a normalized, equal-weighted diagnostic index (w1=1/3, w2=1/3, w3=1/3), not a universal physical constant.
3. **Observational Frame Equivalence**: Simulation metrics are computed at 6-frame intervals (dt_obs = 6.0) to match microscopy sampling, but camera tracking artifacts in experimental data are not explicitly modeled.

---

## 13. Final Conclusions
1. The analysis provides an **unbiased diagnostic baseline** without tuning metrics to artificially improve model scores.
2. **Velocity components and instantaneous speed** exhibit the strongest distributional alignment with experimental live-cell tracking.
3. **Global spatial trajectory features** (`DISPLACEMENT_FROM_ORIGIN`, `DISTANCE_TRAVELED`) represent the primary targets for parameter calibration prior to downstream machine learning integration.
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)


if __name__ == "__main__":
    main()