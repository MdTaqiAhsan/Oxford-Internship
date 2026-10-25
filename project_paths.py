"""
project_paths.py
================================================================================
Central file locations for the Oxford-Internship pipeline.

Every path is built relative to this folder, so all scripts work no matter
which directory they are launched from. Generated results never land in the
project root; each run gets its own timestamped folder under results/:

  results/
  ├── tuning/
  │   ├── optimal_parameters_killing_0to5hrs.json      <- current best (used by the simulator)
  │   ├── optimal_parameters_non-killing_0to5hrs.json
  │   └── run_YYYYMMDD_HHMMSS/                          <- one folder per tuning run
  ├── simulations/run_YYYYMMDD_HHMMSS/                  <- simulated kinematics CSVs
  ├── comparisons/<simulation run>/                     <- metrics + plots for that run
  └── track_windows/                                    <- cached experimental track start/end frames
================================================================================
"""

import os
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

REGIMES = ("killing", "non-killing")
CELL_TYPES = ("tcell", "cancer")

# Raw tracking exports (input of Kinematic_Data_Extractor.py)
RAW_DATA = {
    "killing": {
        "tcell": os.path.join(PROJECT_DIR, "cyto_2023_11_21_tcell_04.csv"),
        "cancer": os.path.join(PROJECT_DIR, "cyto_2023_11_21_ccell_04.csv"),
    },
    "non-killing": {
        "tcell": os.path.join(PROJECT_DIR, "wt_2023_11_18_tcell_04.csv"),
        "cancer": os.path.join(PROJECT_DIR, "wt_2023_11_18_ccell_04.csv"),
    },
}

# Extracted kinematics in physical units (output of Kinematic_Data_Extractor.py)
EXPERIMENTAL_DATA = {
    "killing": {
        "tcell": os.path.join(PROJECT_DIR, "Cyto_T-Cell Kinematics.csv"),
        "cancer": os.path.join(PROJECT_DIR, "Cyto_Cancer Cell Kinematics.csv"),
    },
    "non-killing": {
        "tcell": os.path.join(PROJECT_DIR, "Wt_T-Cell Kinematics.csv"),
        "cancer": os.path.join(PROJECT_DIR, "Wt_Cancer Cell Kinematics.csv"),
    },
}

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TUNING_DIR = os.path.join(RESULTS_DIR, "tuning")
SIMULATIONS_DIR = os.path.join(RESULTS_DIR, "simulations")
COMPARISONS_DIR = os.path.join(RESULTS_DIR, "comparisons")
TRACK_WINDOWS_DIR = os.path.join(RESULTS_DIR, "track_windows")


def optimal_params_path(regime):
    """Current best parameters for a regime (written by tune_parameters.py)."""
    return os.path.join(TUNING_DIR, f"optimal_parameters_{regime}_0to5hrs.json")


def simulation_csv_path(sim_dir, regime, cell_type):
    """Kinematics CSV written by CellSimulation.export_kinematics_csv()."""
    name = "T-cell" if cell_type == "tcell" else "Cancer-cell"
    return os.path.join(sim_dir, f"{regime}_{name}_kinematics.csv")


def new_run_dir(parent):
    """Create and return a fresh results/<kind>/run_YYYYMMDD_HHMMSS folder."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(parent, f"run_{stamp}")
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(parent, f"run_{stamp}_{suffix}")
        suffix += 1
    os.makedirs(path)
    return path


def latest_run_dir(parent):
    """Most recent run_* folder under ``parent``."""
    runs = sorted(
        d for d in os.listdir(parent) if d.startswith("run_") and os.path.isdir(os.path.join(parent, d))
    ) if os.path.isdir(parent) else []
    if not runs:
        raise FileNotFoundError(f"No run_* folders found in {parent}")
    return os.path.join(parent, runs[-1])
