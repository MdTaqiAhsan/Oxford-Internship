import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Experimental datasets
# ------------------------------------------------------------

EXPERIMENTAL_CANCER = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\Cancer Cell Kinematics.csv"
)

EXPERIMENTAL_IMMUNE = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\T-Cell Kinematics.csv"
)

# ------------------------------------------------------------
# Simulation datasets
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = (
    r"C:\Users\taqio\OneDrive\Desktop\CSE\Oxford Internship"
    r"\Oxford-Internship\kinematic_comparison_plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FEATURES
# ============================================================

# FRAME is intentionally excluded from distribution comparison.
# It is a temporal index rather than a kinematic property.

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

# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path, name):
    print(f"\nLoading {name}:")
    print(path)

    df = pd.read_csv(path)

    print(f"Rows: {len(df):,}")

    missing = [
        col for col in KINEMATIC_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing columns:\n{missing}"
        )

    return df


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_feature(exp_values, sim_values):
    """
    Standardize experimental and simulation values together.

    This is deliberately done using the combined distribution so
    both datasets are placed in the same standardized coordinate
    system.

    IMPORTANT:
    This is for distribution comparison, NOT physical calibration.
    """

    exp_values = np.asarray(exp_values).reshape(-1, 1)
    sim_values = np.asarray(sim_values).reshape(-1, 1)

    combined = np.concatenate(
        [exp_values, sim_values],
        axis=0
    )

    scaler = StandardScaler()
    scaler.fit(combined)

    exp_scaled = scaler.transform(exp_values).ravel()
    sim_scaled = scaler.transform(sim_values).ravel()

    return exp_scaled, sim_scaled


# ============================================================
# PLOT ONE COMPARISON
# ============================================================

def plot_comparison(
    experimental_df,
    simulation_df,
    experimental_name,
    simulation_name,
    output_filename
):

    n_features = len(KINEMATIC_COLUMNS)

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 4.8 * nrows)
    )

    axes = np.asarray(axes).flatten()

    # --------------------------------------------------------
    # Plot each feature
    # --------------------------------------------------------

    for i, feature in enumerate(KINEMATIC_COLUMNS):

        ax = axes[i]

        exp = pd.to_numeric(
            experimental_df[feature],
            errors="coerce"
        ).dropna().values

        sim = pd.to_numeric(
            simulation_df[feature],
            errors="coerce"
        ).dropna().values

        # Remove infinities
        exp = exp[np.isfinite(exp)]
        sim = sim[np.isfinite(sim)]

        if len(exp) == 0 or len(sim) == 0:
            ax.set_title(feature + "\n(no valid data)")
            continue

        # ----------------------------------------------------
        # Standardize both datasets together
        # ----------------------------------------------------

        exp_scaled, sim_scaled = standardize_feature(
            exp,
            sim
        )

        # ----------------------------------------------------
        # Histogram
        # ----------------------------------------------------

        ax.hist(
            exp_scaled,
            bins=60,
            density=True,
            alpha=0.55,
            color="blue",
            label="Experimental"
        )

        ax.hist(
            sim_scaled,
            bins=60,
            density=True,
            alpha=0.55,
            color="red",
            label="Simulation"
        )

        ax.set_title(feature, fontsize=11)
        ax.set_xlabel("Standardized value")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend()

    # --------------------------------------------------------
    # Remove unused axes
    # --------------------------------------------------------

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        f"{experimental_name} vs {simulation_name}\n"
        "Kinematic Distribution Comparison",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nSaved:")
    print(output_path)


# ============================================================
# LOAD ALL DATASETS
# ============================================================

experimental_cancer = load_dataset(
    EXPERIMENTAL_CANCER,
    "Experimental Cancer"
)

experimental_immune = load_dataset(
    EXPERIMENTAL_IMMUNE,
    "Experimental Immune"
)

killing_cancer = load_dataset(
    KILLING_CANCER,
    "Killing Cancer"
)

killing_tcell = load_dataset(
    KILLING_TCELL,
    "Killing T-cell"
)

nonkilling_cancer = load_dataset(
    NONKILLING_CANCER,
    "Non-killing Cancer"
)

nonkilling_tcell = load_dataset(
    NONKILLING_TCELL,
    "Non-killing T-cell"
)


# ============================================================
# FOUR COMPARISONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING COMPARISON PLOTS")
print("=" * 70)

# ------------------------------------------------------------
# 1. Experimental Cancer vs Killing Cancer
# ------------------------------------------------------------

plot_comparison(
    experimental_cancer,
    killing_cancer,
    "Experimental Cancer",
    "Killing Simulation Cancer",
    "01_experimental_vs_killing_cancer.png"
)

# ------------------------------------------------------------
# 2. Experimental Cancer vs Non-Killing Cancer
# ------------------------------------------------------------

plot_comparison(
    experimental_cancer,
    nonkilling_cancer,
    "Experimental Cancer",
    "Non-Killing Simulation Cancer",
    "02_experimental_vs_non_killing_cancer.png"
)

# ------------------------------------------------------------
# 3. Experimental Immune vs Killing T-cell
# ------------------------------------------------------------

plot_comparison(
    experimental_immune,
    killing_tcell,
    "Experimental Immune",
    "Killing Simulation T-cell",
    "03_experimental_vs_killing_tcell.png"
)

# ------------------------------------------------------------
# 4. Experimental Immune vs Non-Killing T-cell
# ------------------------------------------------------------

plot_comparison(
    experimental_immune,
    nonkilling_tcell,
    "Experimental Immune",
    "Non-Killing Simulation T-cell",
    "04_experimental_vs_non_killing_tcell.png"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ALL COMPARISONS COMPLETE")
print("=" * 70)
print(f"Plots saved to:")
print(OUTPUT_DIR)

