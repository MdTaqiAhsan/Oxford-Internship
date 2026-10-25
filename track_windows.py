"""
track_windows.py
================================================================================
Experimental track observation windows, used to give simulated cells the same
staggered tracking start/end times as the real data.

In the experiments most tracks do not start at frame 0: cells enter the field
of view or are first detected later, and DT_ACC / DISTANCE_TRAVELED /
DISPLACEMENT_FROM_ORIGIN are measured from each track's own start. Simulated
cells therefore get an observation window [start_frame, end_frame] sampled from
the real tracks, and their kinematics are recorded relative to start_frame.

Also estimates the real cell population (peak number of tracked objects per
frame in the raw data, short tracks included), used to size the simulation.
================================================================================
"""

import os

import numpy as np
import pandas as pd

from project_paths import EXPERIMENTAL_DATA, RAW_DATA, TRACK_WINDOWS_DIR

SECONDS_PER_FRAME = 10.0
CHUNK_SIZE = 2_000_000


def _cache_path(regime, cell_type, kind):
    return os.path.join(TRACK_WINDOWS_DIR, f"{regime}_{cell_type}_{kind}.csv")


def _is_fresh(cache, src):
    return os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src)


def _scan_track_windows(path):
    """Start and end frame of every track in an extracted kinematics CSV."""
    starts, ends = {}, {}
    for chunk in pd.read_csv(path, usecols=["TRACK_ID_TA", "FRAME", "DT_ACC"], chunksize=CHUNK_SIZE):
        # DT_ACC (seconds) is the time since the track started, so the true start
        # frame is FRAME - DT_ACC / 10 even when the first row is not at a multiple of 6.
        chunk = chunk.drop_duplicates(subset=["TRACK_ID_TA", "FRAME"])
        chunk["START"] = chunk["FRAME"] - chunk["DT_ACC"] / SECONDS_PER_FRAME
        grp = chunk.groupby("TRACK_ID_TA")
        for tid, s in grp["START"].min().items():
            starts[tid] = min(s, starts.get(tid, s))
        for tid, e in grp["FRAME"].max().items():
            ends[tid] = max(e, ends.get(tid, e))

    windows = pd.DataFrame({"start_frame": pd.Series(starts), "end_frame": pd.Series(ends)})
    windows["start_frame"] = windows["start_frame"].round().clip(lower=0).astype(int)
    windows["end_frame"] = windows["end_frame"].astype(int)
    windows.index.name = "TRACK_ID_TA"
    return windows


def _scan_frame_counts(path):
    """Number of tracked objects per FRAME in a raw tracking CSV (all tracks, short ones included)."""
    counts = None
    for chunk in pd.read_csv(path, usecols=["TRACK_ID_TA", "FRAME"], chunksize=CHUNK_SIZE):
        c = chunk.drop_duplicates()["FRAME"].value_counts()
        counts = c if counts is None else counts.add(c, fill_value=0)
    counts = counts.sort_index().astype(int)
    counts.index.name = "FRAME"
    counts.name = "tracked_cells"
    return counts


def load_track_windows(regime, cell_type):
    """DataFrame (index TRACK_ID_TA) with start_frame and end_frame of every experimental track."""
    src = EXPERIMENTAL_DATA[regime][cell_type]
    cache = _cache_path(regime, cell_type, "windows")
    if _is_fresh(cache, src):
        return pd.read_csv(cache, index_col="TRACK_ID_TA")
    print(f"[Track windows] Scanning {os.path.basename(src)} (cached afterwards)...")
    windows = _scan_track_windows(src)
    os.makedirs(TRACK_WINDOWS_DIR, exist_ok=True)
    windows.to_csv(cache)
    return windows


def load_frame_counts(regime, cell_type):
    """Series: number of tracked objects per FRAME in the raw data."""
    src = RAW_DATA[regime][cell_type]
    cache = _cache_path(regime, cell_type, "frame_counts")
    if _is_fresh(cache, src):
        return pd.read_csv(cache, index_col="FRAME")["tracked_cells"]
    print(f"[Track windows] Counting cells per frame in {os.path.basename(src)} (cached afterwards)...")
    counts = _scan_frame_counts(src)
    os.makedirs(TRACK_WINDOWS_DIR, exist_ok=True)
    counts.to_csv(cache)
    return counts


def estimate_population(regime, cell_type, frame_max, density_fraction=1.0):
    """Simulated population size for a cell type.

    Uses the median number of simultaneously tracked objects per frame in
    frames 0..frame_max of the raw data as the estimate of the real population,
    scaled by ``density_fraction`` (e.g. 0.5 = half the real density, to save compute).
    """
    counts = load_frame_counts(regime, cell_type)
    typical = float(counts[counts.index <= frame_max].median())
    return max(1, int(round(typical * density_fraction)))


def sample_windows(windows, n, horizon, rng):
    """Sample ``n`` observation windows (start, end frame arrays) for simulated cells.

    Only tracks that start before ``horizon`` (the simulated number of frames)
    can be observed in the simulation, so only those are sampled.
    """
    eligible = windows[windows["start_frame"] < horizon]
    if eligible.empty:
        raise ValueError(f"No experimental tracks start before frame {horizon}.")
    idx = rng.integers(0, len(eligible), size=n)
    picked = eligible.iloc[idx]
    return picked["start_frame"].to_numpy(dtype=np.int64), picked["end_frame"].to_numpy(dtype=np.int64)


def sample_population_windows(regime, num_immune, num_cancer, horizon, seed):
    """Observation windows for a whole simulated population (immune cells first, then cancer)."""
    rng = np.random.default_rng(seed)
    t_start, t_end = sample_windows(load_track_windows(regime, "tcell"), num_immune, horizon, rng)
    c_start, c_end = sample_windows(load_track_windows(regime, "cancer"), num_cancer, horizon, rng)
    return np.concatenate([t_start, c_start]), np.concatenate([t_end, c_end])
