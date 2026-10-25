"""
Kinematic_Data_Extractor.py
================================================================================
Turns the raw tracking exports (cyto_* = killing, wt_* = non-killing) into the
"<Regime>_<Cell> Kinematics.csv" files used by every other script:
  - keeps whole tracks lasting >= 1 h (T-cells) / >= 15 min (cancer cells)
  - converts pixels -> µm, pixel/frame -> µm/s, frames -> seconds
  - ignores zero-length steps: rows where the cell did not move since the
    previous frame (DISPLACEMENT_FROM_PREVIOUS_POINT == 0, including the first
    row of each track) get NaN step features (DX/DY/DISPLACEMENT_FROM_PREVIOUS_POINT,
    VEL_X, VEL_Y, SPEED). The simulation never produces exactly-zero steps, so
    these would otherwise form a spike it cannot match. The row's cumulative
    features (DISTANCE_TRAVELED, DISPLACEMENT_FROM_ORIGIN, ...) are kept.
Input and output paths come from project_paths.py.
================================================================================
"""

import os
import numpy as np
import pandas as pd

from project_paths import EXPERIMENTAL_DATA, RAW_DATA

# ============================================================
# CONFIGURATION & THRESHOLDS
# ============================================================

# Minimum trajectory duration thresholds (in raw DT_ACC units, i.e., frames)
CANCER_MIN_DT_ACC = 90    # ~15 minutes (90 frames)
IMMUNE_MIN_DT_ACC = 360   # ~1 hour (360 frames)

# Primary Track Identifier
TRACK_ID_COLUMN = "TRACK_ID_TA"

# ------------------------------------------------------------
# INPUT DATASETS
# ------------------------------------------------------------
CYTO_T_CELL_INPUT = RAW_DATA["killing"]["tcell"]
CYTO_CANCER_CELL_INPUT = RAW_DATA["killing"]["cancer"]

WT_T_CELL_INPUT = RAW_DATA["non-killing"]["tcell"]
WT_CANCER_CELL_INPUT = RAW_DATA["non-killing"]["cancer"]

# ------------------------------------------------------------
# OUTPUT DATASETS
# ------------------------------------------------------------
CYTO_T_CELL_OUTPUT = EXPERIMENTAL_DATA["killing"]["tcell"]
CYTO_CANCER_CELL_OUTPUT = EXPERIMENTAL_DATA["killing"]["cancer"]

WT_T_CELL_OUTPUT = EXPERIMENTAL_DATA["non-killing"]["tcell"]
WT_CANCER_CELL_OUTPUT = EXPERIMENTAL_DATA["non-killing"]["cancer"]

# ------------------------------------------------------------
# PROCESSING & CONVERSION PARAMETERS
# ------------------------------------------------------------
CHUNK_SIZE = 100_000

# 1 pixel = 1.6 micrometres
PIXEL_TO_MICROMETER = 1.6

# 1 frame interval = 10.0 seconds
SECONDS_PER_FRAME_INTERVAL = 10.0


# ============================================================
# KINEMATIC COLUMNS
# ============================================================

KINEMATIC_COLUMNS = [
    "TRACK_ID_TA",
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
    "DT_ACC",
    "DX_ACC",
    "DY_ACC",
]

# Spatial measurements: pixel -> micrometre (× 1.6)
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
    "DX_ACC",
    "DY_ACC",
]

# Velocity/Speed: pixel/frame -> µm/s (× 1.6 ÷ 10 = × 0.16)
VELOCITY_COLUMNS = [
    "VEL_X",
    "VEL_Y",
    "SPEED",
    "AVERAGE_SPEED",
]

# Time: frames -> seconds (× 10.0)
TIME_COLUMNS = [
    "DT_ACC",
]

# Per-step features that are meaningless for a zero-length step (set to NaN)
STEP_COLUMNS = [
    "DX_FROM_PREVIOUS_POINT",
    "DY_FROM_PREVIOUS_POINT",
    "DISPLACEMENT_FROM_PREVIOUS_POINT",
    "VEL_X",
    "VEL_Y",
    "SPEED",
]


# ============================================================
# TRAJECTORY FILTERING & CONVERSION ENGINE
# ============================================================

def extract_and_filter_kinematics(
    input_file,
    output_file,
    min_dt_acc_threshold,
    cell_type_label,
    conversion_factor=PIXEL_TO_MICROMETER,
    seconds_per_frame=SECONDS_PER_FRAME_INTERVAL
):
    print("\n" + "=" * 75)
    print(f"Reading: {input_file} ({cell_type_label})")
    print(f"Output:  {output_file}")
    print(f"Filter:  Retaining {TRACK_ID_COLUMN} tracks with max(DT_ACC) >= {min_dt_acc_threshold} frames")
    print("=" * 75)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist:\n{input_file}")

    # Check header for required columns
    header = pd.read_csv(input_file, nrows=0)
    missing = [col for col in KINEMATIC_COLUMNS if col not in header.columns]
    if missing:
        raise ValueError(
            f"\nMissing columns in {input_file}:\n" + "\n".join(missing)
        )

    # ------------------------------------------------------------
    # PASS 1: Calculate max(DT_ACC) per TRACK_ID_TA (Chunked)
    # ------------------------------------------------------------
    print("\n[Pass 1/2] Scanning trajectories to determine max(DT_ACC)...")
    track_max_dt = {}

    for chunk in pd.read_csv(input_file, usecols=[TRACK_ID_COLUMN, "DT_ACC"], chunksize=CHUNK_SIZE):
        chunk["DT_ACC"] = pd.to_numeric(chunk["DT_ACC"], errors="coerce")
        grouped = chunk.groupby(TRACK_ID_COLUMN)["DT_ACC"].max()

        for track_id, max_val in grouped.items():
            if pd.notnull(max_val):
                if track_id not in track_max_dt:
                    track_max_dt[track_id] = max_val
                else:
                    track_max_dt[track_id] = max(track_max_dt[track_id], max_val)

    # Set of tracks that meet or exceed the duration threshold
    qualifying_tracks = {
        track_id for track_id, max_dt in track_max_dt.items() 
        if max_dt >= min_dt_acc_threshold
    }

    total_tracks = len(track_max_dt)
    retained_tracks = len(qualifying_tracks)
    excluded_tracks = total_tracks - retained_tracks

    print(f"Total Unique Tracks Scanned:    {total_tracks:,}")
    print(f"Tracks Excluded (duration < {min_dt_acc_threshold}): {excluded_tracks:,}")
    retained_pct = (retained_tracks / total_tracks * 100) if total_tracks > 0 else 0.0
    print(f"Tracks Retained (max DT >= {min_dt_acc_threshold}):  {retained_tracks:,} ({retained_pct:.2f}%)")

    # ------------------------------------------------------------
    # PASS 2: Filter and Convert Qualifying Trajectories
    # ------------------------------------------------------------
    print("\n[Pass 2/2] Filtering and applying physical unit conversions...")
    # Remove any output from a previous run, so a run that retains nothing
    # cannot leave a stale file behind
    if os.path.exists(output_file):
        os.remove(output_file)

    first_chunk = True
    total_retained_rows = 0
    total_zero_steps = 0

    for chunk in pd.read_csv(input_file, usecols=KINEMATIC_COLUMNS, chunksize=CHUNK_SIZE):
        # Keep entire trajectories of qualifying tracks only
        filtered_chunk = chunk[chunk[TRACK_ID_COLUMN].isin(qualifying_tracks)].copy()

        if len(filtered_chunk) == 0:
            continue

        # 1. Convert spatial quantities: pixel -> micrometre (× 1.6)
        for col in SPATIAL_COLUMNS:
            filtered_chunk[col] = (
                pd.to_numeric(filtered_chunk[col], errors="coerce") * conversion_factor
            )

        # 2. Convert velocity/speed: pixel/frame -> µm/s (× 1.6 ÷ 10 = × 0.16)
        for col in VELOCITY_COLUMNS:
            filtered_chunk[col] = (
                pd.to_numeric(filtered_chunk[col], errors="coerce")
                * conversion_factor
                / seconds_per_frame
            )

        # 3. Convert accumulated tracking time: frames -> seconds (× 10.0)
        for col in TIME_COLUMNS:
            filtered_chunk[col] = (
                pd.to_numeric(filtered_chunk[col], errors="coerce")
                * seconds_per_frame
            )

        # 4. Ignore zero-length steps (see module docstring)
        zero_step = filtered_chunk["DISPLACEMENT_FROM_PREVIOUS_POINT"] == 0
        filtered_chunk.loc[zero_step, STEP_COLUMNS] = np.nan
        total_zero_steps += int(zero_step.sum())

        # Write to final CSV
        filtered_chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        total_retained_rows += len(filtered_chunk)
        first_chunk = False

        print(f"\rProcessed and wrote {total_retained_rows:,} qualifying rows", end="")

    print(f"\n\nFinished: {output_file}")
    print(f"Total rows retained for analysis: {total_retained_rows:,}")
    if total_retained_rows:
        print(f"Zero-length steps ignored (step features set to NaN): {total_zero_steps:,} "
              f"({total_zero_steps / total_retained_rows * 100:.1f}% of rows)")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # 1. CYTO / KILLING T-CELL (Threshold >= 360 frames)
    extract_and_filter_kinematics(
        CYTO_T_CELL_INPUT,
        CYTO_T_CELL_OUTPUT,
        min_dt_acc_threshold=IMMUNE_MIN_DT_ACC,
        cell_type_label="Immune / T-Cell"
    )

    # 2. CYTO / KILLING CANCER (Threshold >= 90 frames)
    extract_and_filter_kinematics(
        CYTO_CANCER_CELL_INPUT,
        CYTO_CANCER_CELL_OUTPUT,
        min_dt_acc_threshold=CANCER_MIN_DT_ACC,
        cell_type_label="Cancer Cell"
    )

    # 3. WT / NON-KILLING T-CELL (Threshold >= 360 frames)
    extract_and_filter_kinematics(
        WT_T_CELL_INPUT,
        WT_T_CELL_OUTPUT,
        min_dt_acc_threshold=IMMUNE_MIN_DT_ACC,
        cell_type_label="Immune / T-Cell"
    )

    # 4. WT / NON-KILLING CANCER (Threshold >= 90 frames)
    extract_and_filter_kinematics(
        WT_CANCER_CELL_INPUT,
        WT_CANCER_CELL_OUTPUT,
        min_dt_acc_threshold=CANCER_MIN_DT_ACC,
        cell_type_label="Cancer Cell"
    )

    print("\n" + "=" * 75)
    print("KINEMATIC EXTRACTION & TRAJECTORY-DURATION FILTERING COMPLETE")
    print("=" * 75)
    print("All generated datasets strictly retain complete trajectories where:")
    print(f"  - Cancer Cells:   max(DT_ACC) >= {CANCER_MIN_DT_ACC} frames (~15 min)")
    print(f"  - Immune T-Cells: max(DT_ACC) >= {IMMUNE_MIN_DT_ACC} frames (~1 hour)")
    print("=" * 75)