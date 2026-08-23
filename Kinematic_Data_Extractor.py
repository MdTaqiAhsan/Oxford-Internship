import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# INPUT DATASETS
# ------------------------------------------------------------

CYTO_T_CELL_INPUT = "cyto_2023_11_21_tcell_04.csv"
CYTO_CANCER_CELL_INPUT = "cyto_2023_11_21_ccell_04.csv"

WT_T_CELL_INPUT = "wt_2023_11_18_tcell_04.csv"
WT_CANCER_CELL_INPUT = "wt_2023_11_18_ccell_04.csv"

# ------------------------------------------------------------
# OUTPUT DATASETS
# ------------------------------------------------------------

CYTO_T_CELL_OUTPUT = "Cyto_T-Cell Kinematics.csv"
CYTO_CANCER_CELL_OUTPUT = "Cyto_Cancer Cell Kinematics.csv"

WT_T_CELL_OUTPUT = "Wt_T-Cell Kinematics.csv"
WT_CANCER_CELL_OUTPUT = "Wt_Cancer Cell Kinematics.csv"

# ------------------------------------------------------------
# PROCESSING CONFIGURATION
# ------------------------------------------------------------

CHUNK_SIZE = 100_000

# ------------------------------------------------------------
# UNIT CONVERSION
# ------------------------------------------------------------

# Spatial measurements:
#   pixel -> micrometre
#
# Assumption:
#   1 pixel = 1.6 micrometres

PIXEL_TO_MICROMETER = 1.6

# Velocity/speed measurements in the experimental files
# are assumed to be:
#
#   pixel/frame
#
# One frame interval represents:
#
#   10 seconds
#
# Therefore:
#
#   pixel/frame
#       × 1.6 µm/pixel
#       ÷ 10 seconds/frame
#       =
#   µm/second
#
SECONDS_PER_FRAME_INTERVAL = 10.0


# ============================================================
# KINEMATIC COLUMNS
# ============================================================

KINEMATIC_COLUMNS = [
    "FRAME",

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
# SPATIAL COLUMNS
# ============================================================

# These quantities are distances.
#
# Experimental units:
#     pixel
#
# Output units:
#     micrometre
#
# Conversion:
#     pixel × 1.6 = micrometre

SPATIAL_COLUMNS = [
    "POSITION_X",
    "POSITION_Y",

    "DX_FROM_PREVIOUS_POINT",
    "DY_FROM_PREVIOUS_POINT",
    "DISPLACEMENT_FROM_PREVIOUS_POINT",

    "DX_FROM_ORIGIN",
    "DY_FROM_ORIGIN",
    "DISPLACEMENT_FROM_ORIGIN",

    "DISTANCE_TRAVELED",
]


# ============================================================
# VELOCITY / SPEED COLUMNS
# ============================================================

# These quantities are assumed to be given as:
#
#     pixel/frame
#
# We convert them to:
#
#     micrometre/second
#
# using:
#
#     pixel/frame × 1.6 µm/pixel ÷ 10 s/frame
#
# Net multiplication:
#
#     × 0.16

VELOCITY_COLUMNS = [
    "VEL_X",
    "VEL_Y",
    "SPEED",
    "AVERAGE_SPEED",
]


# ============================================================
# EXTRACT + CONVERT KINEMATIC DATA
# ============================================================

def extract_kinematics(
    input_file,
    output_file,
    conversion_factor=PIXEL_TO_MICROMETER,
    seconds_per_frame=SECONDS_PER_FRAME_INTERVAL
):

    print("\n" + "=" * 70)
    print(f"Reading: {input_file}")
    print(f"Output:  {output_file}")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input file exists
    # --------------------------------------------------------

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file does not exist:\n{input_file}"
        )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    header = pd.read_csv(
        input_file,
        nrows=0
    )

    missing = [
        col
        for col in KINEMATIC_COLUMNS
        if col not in header.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing columns in {input_file}:\n"
            + "\n".join(missing)
        )

    print(
        f"All {len(KINEMATIC_COLUMNS)} required "
        "kinematic/frame columns found."
    )

    # --------------------------------------------------------
    # Display conversion information
    # --------------------------------------------------------

    print("\nUnit conversions:")

    print(
        f"  Spatial quantities: "
        f"pixel -> micrometre (× {conversion_factor})"
    )

    print(
        f"  Velocity/speed quantities: "
        f"pixel/frame -> micrometre/second "
        f"(× {conversion_factor} ÷ {seconds_per_frame})"
    )

    net_velocity_factor = (
        conversion_factor / seconds_per_frame
    )

    print(
        f"  Net velocity/speed factor: "
        f"× {net_velocity_factor}"
    )

    print("\nSpatial columns converted:")
    for col in SPATIAL_COLUMNS:
        print(f"  {col}")

    print("\nVelocity/speed columns converted:")
    for col in VELOCITY_COLUMNS:
        print(f"  {col}")

    print("\nUnchanged columns:")
    print("  FRAME")
    print("  PATH_EFFICIENCY")

    # --------------------------------------------------------
    # Process CSV in chunks
    # --------------------------------------------------------

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        input_file,
        usecols=KINEMATIC_COLUMNS,
        chunksize=CHUNK_SIZE
    ):

        # ----------------------------------------------------
        # Convert spatial quantities
        #
        # pixel -> micrometre
        # ----------------------------------------------------

        for col in SPATIAL_COLUMNS:

            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce"
            )

            chunk[col] = (
                chunk[col]
                * conversion_factor
            )

        # ----------------------------------------------------
        # Convert velocity/speed quantities
        #
        # pixel/frame -> micrometre/second
        #
        # pixel/frame × 1.6 ÷ 10
        # ----------------------------------------------------

        for col in VELOCITY_COLUMNS:

            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce"
            )

            chunk[col] = (
                chunk[col]
                * conversion_factor
                / seconds_per_frame
            )

        # ----------------------------------------------------
        # Write converted chunk
        # ----------------------------------------------------

        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        total_rows += len(chunk)
        first_chunk = False

        print(
            f"\rProcessed {total_rows:,} rows",
            end=""
        )

    print(f"\n\nFinished: {output_file}")
    print(f"Total rows: {total_rows:,}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. CYTO / KILLING T-CELL
    # --------------------------------------------------------

    extract_kinematics(
        CYTO_T_CELL_INPUT,
        CYTO_T_CELL_OUTPUT
    )

    # --------------------------------------------------------
    # 2. CYTO / KILLING CANCER
    # --------------------------------------------------------

    extract_kinematics(
        CYTO_CANCER_CELL_INPUT,
        CYTO_CANCER_CELL_OUTPUT
    )

    # --------------------------------------------------------
    # 3. WT / NON-KILLING T-CELL
    # --------------------------------------------------------

    extract_kinematics(
        WT_T_CELL_INPUT,
        WT_T_CELL_OUTPUT
    )

    # --------------------------------------------------------
    # 4. WT / NON-KILLING CANCER
    # --------------------------------------------------------

    extract_kinematics(
        WT_CANCER_CELL_INPUT,
        WT_CANCER_CELL_OUTPUT
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("KINEMATIC EXTRACTION + UNIT CONVERSION COMPLETE")
    print("=" * 70)

    print("\nCreated datasets:")

    print(f"  1. {CYTO_T_CELL_OUTPUT}")
    print(f"  2. {CYTO_CANCER_CELL_OUTPUT}")
    print(f"  3. {WT_T_CELL_OUTPUT}")
    print(f"  4. {WT_CANCER_CELL_OUTPUT}")

    print("\nUnit conversion applied:")

    print(
        "  Distance:       pixel -> micrometre       × 1.6"
    )

    print(
        "  Velocity X:     pixel/frame -> µm/second  × 1.6 / 10"
    )

    print(
        "  Velocity Y:     pixel/frame -> µm/second  × 1.6 / 10"
    )

    print(
        "  Speed:          pixel/frame -> µm/second  × 1.6 / 10"
    )

    print(
        "  Average speed:  pixel/frame -> µm/second  × 1.6 / 10"
    )

    print("\nNet velocity/speed conversion:")
    print("  × 0.16")

    print("\nNot converted:")
    print("  FRAME")
    print("  PATH_EFFICIENCY")

    print("=" * 70)