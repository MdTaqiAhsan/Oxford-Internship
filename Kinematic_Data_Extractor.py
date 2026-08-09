import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

T_CELL_INPUT = "cyto_2023_11_21_tcell_04.csv"
CANCER_CELL_INPUT = "cyto_2023_11_21_ccell_04.csv"

T_CELL_OUTPUT = "T-Cell Kinematics.csv"
CANCER_CELL_OUTPUT = "Cancer Cell Kinematics.csv"

CHUNK_SIZE = 100_000

# FRAME is included because the experimental data is
# sampled at specific time/frame intervals (0, 6, 12, ...).
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
# EXTRACT KINEMATIC DATA
# ============================================================

def extract_kinematics(input_file, output_file):

    print(f"\nReading: {input_file}")
    print(f"Output:  {output_file}")

    # --------------------------------------------------------
    # Check that all required columns exist
    # --------------------------------------------------------

    header = pd.read_csv(input_file, nrows=0)

    missing = [
        col for col in KINEMATIC_COLUMNS
        if col not in header.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {input_file}:\n"
            + "\n".join(missing)
        )

    print(
        f"All {len(KINEMATIC_COLUMNS)} required "
        "kinematic/frame columns found."
    )

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

        # Write header only for the first chunk
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

    print(f"\nFinished: {output_file}")
    print(f"Total rows: {total_rows:,}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    extract_kinematics(
        T_CELL_INPUT,
        T_CELL_OUTPUT
    )

    extract_kinematics(
        CANCER_CELL_INPUT,
        CANCER_CELL_OUTPUT
    )

    print("\n======================================")
    print("KINEMATIC EXTRACTION COMPLETE")
    print("======================================")
    print(f"Created: {T_CELL_OUTPUT}")
    print(f"Created: {CANCER_CELL_OUTPUT}")

