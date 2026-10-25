"""
tune_parameters.py
================================================================================
Bayesian Optimization (Optuna) of the Cell-Motility Simulator
Target Temporal Window: 0 to 5 Hours (Frames 0 to 1800)
--------------------------------------------------------------------------------
For each regime (killing = Cyto data, non-killing = Wt data) this searches the
simulator parameters whose simulated kinematic distributions best match the
experimental ones.

HOW SIMULATED DATA IS MADE COMPARABLE TO THE EXPERIMENT:
  - Staggered tracks: every simulated cell gets an observation window
    (start/end frame) sampled from the real tracks (track_windows.py), and its
    DT_ACC, DISTANCE_TRAVELED, DISPLACEMENT_FROM_ORIGIN, ... are measured from
    that start, exactly like an experimental track that begins mid-movie.
  - Same duration filter on both sides: a track counts if it lasts
    >= 1 h (T-cells) / >= 15 min (cancer cells) over its whole lifetime.
  - Same sampling: experimental rows exist every 6th frame (60 s), so
    simulated rows are taken at FRAME % 6 == 0 too, within FRAME <= 1800.
  - Population: DENSITY_FRACTION x the peak number of tracked cells per frame
    in the experiment (0.5 = half the real density, to save compute).

RANDOM SEEDS:
  - Every trial is scored on the same TRAIN_SEEDS (averaged), so trials are
    compared on equal footing.
  - The TOP_K_VALIDATE best trials are then re-scored on fresh
    VALIDATION_SEEDS, and the trial with the best validation score is
    exported. This avoids picking parameters that only fit one random layout.

OPTIMIZATION FEATURE TARGETS (BALANCED ISOTROPIC OBJECTIVE):
Because cell motility is spatially isotropic, Y-axis coordinates are omitted to
prevent redundant search budget expenditure. However, to avoid sign-cancellation
and preserve 2D path tortuosity, scalar magnitudes and efficiency ratios are retained:

  1. LOCAL STEP MOTILITY (Total: 0.35)
     - DX_FROM_PREVIOUS_POINT (0.10): Governs 1D local step spread.
     - DISPLACEMENT_FROM_PREVIOUS_POINT (0.10): Governs 2D step distance magnitude.
     - SPEED (0.15): Anchors instantaneous scalar velocity.

  2. NET DISPLACEMENT & PERSISTENCE (Total: 0.25)
     - DX_FROM_ORIGIN (0.10): Governs 1D dispersion from starting point.
     - DISPLACEMENT_FROM_ORIGIN (0.15): Anchors Euclidean origin displacement.

  3. CUMULATIVE PATH MILEAGE (Total: 0.25)
     - DX_ACC (0.10): Governs accumulated 1D distance traveled.
     - DISTANCE_TRAVELED (0.15): Anchors resultant 2D path length.

  4. TORTUOSITY & MEANDERING (Total: 0.15)
     - PATH_EFFICIENCY (0.15): Ratio of displacement to total distance.

EXCLUDED FEATURES (Weight = 0.00):
  - DY_FROM_PREVIOUS_POINT, DY_FROM_ORIGIN, DY_ACC: Redundant via spatial isotropy.
  - VEL_X, VEL_Y: Redundant with step displacements scaled by dt = 10s.
  - AVERAGE_SPEED: Redundant with DISTANCE_TRAVELED / DT_ACC.
  - POSITION_X, POSITION_Y: Excluded to prevent spatial pinning.
  - DT_ACC: Governed strictly by frame index and survival thresholds.

OUTPUT (all under results/tuning/):
  run_YYYYMMDD_HHMMSS/
      optimal_parameters_{regime}_0to5hrs.json
      optimization_trials_{regime}_0to5hrs.csv
      best_feature_scores_{regime}_0to5hrs.csv
      run_config.json
  optimal_parameters_{regime}_0to5hrs.json   <- copy of the latest result,
                                                read by cell_simulator.py

Usage:
    python tune_parameters.py                      # both regimes, 30 trials each
    python tune_parameters.py --trials 100 --regimes killing
================================================================================
"""

import argparse
import gc
import json
import os
import shutil

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import optuna

import project_paths as pp
from cell_simulator import CellSimulation
from track_windows import estimate_population, sample_population_windows

# Suppress Optuna trial verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 1. TEMPORAL HORIZON, RESOLUTION & DURATION THRESHOLDS
# =============================================================================
SECONDS_PER_FRAME = 10.0
WINDOW_HOURS = 5.0
FRAME_CUTOFF = int(WINDOW_HOURS * 3600.0 / SECONDS_PER_FRAME)  # 1800 frames
OBSERVATION_STRIDE = 6  # experimental rows are every 6th frame (60 s)

CANCER_MIN_DT_ACC_SECONDS = 900.0   # 15 min (90 raw frames * 10s)
IMMUNE_MIN_DT_ACC_SECONDS = 3600.0  # 1 hr (360 raw frames * 10s)

NUM_TRIALS = 30                      # Bayesian optimization trials per regime
TRAIN_SEEDS = (1001,)                # every trial is scored on these (averaged)
VALIDATION_SEEDS = (3003, 4004)      # fresh seeds to re-check the best trials
TOP_K_VALIDATE = 3
SAMPLER_SEED = 42                    # makes the Optuna search itself reproducible

DENSITY_FRACTION = 0.5               # simulate half the real cell density (compute budget)

# =============================================================================
# 2. FEATURE SPECIFICATIONS & BALANCED WEIGHTS
# =============================================================================
FEATURES_17 = [
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

# Total sum = 1.00
OPTIMIZED_FEATURE_WEIGHTS = {
    # 1. Local Step Motility (Total: 0.35)
    "DX_FROM_PREVIOUS_POINT":           0.10,
    "DISPLACEMENT_FROM_PREVIOUS_POINT": 0.10,
    "SPEED":                            0.15,
    "DY_FROM_PREVIOUS_POINT":           0.00,
    "VEL_X":                            0.00,
    "VEL_Y":                            0.00,

    # 2. Net Displacement & Persistence (Total: 0.25)
    "DX_FROM_ORIGIN":                   0.10,
    "DISPLACEMENT_FROM_ORIGIN":         0.15,
    "DY_FROM_ORIGIN":                   0.00,

    # 3. Cumulative Path Mileage (Total: 0.25)
    "DX_ACC":                           0.10,
    "DISTANCE_TRAVELED":                0.15,
    "DY_ACC":                           0.00,
    "AVERAGE_SPEED":                    0.00,

    # 4. Path Tortuosity / Meandering (Total: 0.15)
    "PATH_EFFICIENCY":                  0.15,

    # 5. Boundary / Temporal (Total: 0.00)
    "POSITION_X":                       0.00,
    "POSITION_Y":                       0.00,
    "DT_ACC":                           0.00,
}

NON_NEGATIVE_FEATURES = {
    "DISPLACEMENT_FROM_PREVIOUS_POINT",
    "DISPLACEMENT_FROM_ORIGIN",
    "DX_ACC",
    "DY_ACC",
    "DT_ACC",
    "DISTANCE_TRAVELED",
    "PATH_EFFICIENCY",
    "SPEED",
    "AVERAGE_SPEED",
}


def clean_values(feat, vals):
    vals = np.asarray(vals, dtype=np.float64)
    valid_mask = np.isfinite(vals)
    if feat in NON_NEGATIVE_FEATURES:
        valid_mask = valid_mask & (vals >= 0.0)
    if feat == "PATH_EFFICIENCY":
        valid_mask = valid_mask & (vals <= 1.05)
    return vals[valid_mask]


# =============================================================================
# 3. EXPERIMENTAL DATA LOADER & VALIDATOR
# =============================================================================
def load_and_validate_experimental_data(filepath, is_cancer=False):
    cell_type = "Cancer" if is_cancer else "Immune/T-Cell"
    min_dur_seconds = CANCER_MIN_DT_ACC_SECONDS if is_cancer else IMMUNE_MIN_DT_ACC_SECONDS

    print(f"[Loading Experimental Reference] {cell_type} Dataset:")
    print(f"  -> {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required experimental file: {filepath}")

    header = pd.read_csv(filepath, nrows=0)
    track_col = "TRACK_ID_TA"
    if track_col not in header.columns:
        raise KeyError(f"Critical Error: Primary identifier '{track_col}' not found in {filepath}!")

    missing_cols = [c for c in FEATURES_17 if c not in header.columns]
    if missing_cols:
        raise KeyError(f"Critical Error: Missing required features in {filepath}:\n{missing_cols}")

    df = pd.read_csv(filepath, usecols=[track_col, "FRAME"] + FEATURES_17)

    # Drop duplicate (track, frame) records
    duplicates = int(df.duplicated(subset=[track_col, "FRAME"]).sum())
    if duplicates > 0:
        print(f"  [Notice] Dropping {duplicates} duplicate ({track_col}, FRAME) records.")
        df = df.drop_duplicates(subset=[track_col, "FRAME"])

    # Whole-track duration filter (the simulation applies the same rule)
    track_max_dt = df.groupby(track_col)["DT_ACC"].max()
    qualifying_tracks = set(track_max_dt[track_max_dt >= min_dur_seconds].index)

    total_tracks = len(track_max_dt)
    retained_tracks = len(qualifying_tracks)
    retention_pct = (retained_tracks / total_tracks * 100.0) if total_tracks > 0 else 0.0

    print(f"  • Total Tracks: {total_tracks:,} | Retained: {retained_tracks:,} ({retention_pct:.2f}%) [Threshold >= {min_dur_seconds:.0f}s]")

    df_qualified = df[df[track_col].isin(qualifying_tracks)]
    df_windowed = df_qualified[df_qualified["FRAME"] <= FRAME_CUTOFF]
    print(f"  • Observations within Horizon (0 to 5h): {len(df_windowed):,} rows (FRAME <= {FRAME_CUTOFF})\n")

    return {feat: clean_values(feat, pd.to_numeric(df_windowed[feat], errors="coerce").values)
            for feat in FEATURES_17}


# =============================================================================
# 4. STATISTICAL METRIC COMPUTATION
# =============================================================================
def compute_metrics_for_feature(exp_arr, sim_arr):
    if len(exp_arr) < 5 or len(sim_arr) < 5:
        return 1.0, 999.0, 999.0, 1.0

    # 1. Kolmogorov-Smirnov Statistic
    try:
        ks_res = stats.ks_2samp(exp_arr, sim_arr)
        ks_stat = float(ks_res.statistic)
    except Exception:
        ks_stat = 1.0

    # 2. Raw Wasserstein-1 Distance
    try:
        w1_raw = float(stats.wasserstein_distance(exp_arr, sim_arr))
    except Exception:
        w1_raw = 999.0

    # 3. Standardized Wasserstein-1 Distance (Experimental Reference Z-Scale)
    e_mean, e_std = np.mean(exp_arr), np.std(exp_arr)
    if e_std > 1e-6:
        exp_z = (exp_arr - e_mean) / e_std
        sim_z = (sim_arr - e_mean) / e_std
        try:
            w1_std = float(stats.wasserstein_distance(exp_z, sim_z))
        except Exception:
            w1_std = 999.0
    else:
        w1_std = w1_raw

    # 4. Jensen-Shannon Divergence
    try:
        z_min, z_max, num_bins = -8.0, 8.0, 100
        e_z_clamp = np.clip((exp_arr - e_mean) / (e_std + 1e-8), z_min, z_max)
        s_z_clamp = np.clip((sim_arr - e_mean) / (e_std + 1e-8), z_min, z_max)
        bins = np.linspace(z_min, z_max, num_bins + 1)

        p, _ = np.histogram(e_z_clamp, bins=bins, density=False)
        q, _ = np.histogram(s_z_clamp, bins=bins, density=False)

        p = (p + 1e-12) / (np.sum(p) + 1e-12 * num_bins)
        q = (q + 1e-12) / (np.sum(q) + 1e-12 * num_bins)
        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log2(p / m))
        kl_qm = np.sum(q * np.log2(q / m))
        jsd = float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))
    except Exception:
        jsd = 1.0

    return ks_stat, w1_raw, w1_std, jsd


# =============================================================================
# 5. SIMULATION INSTANCE RUNNER
# =============================================================================
def run_simulation_instance(p, mode, seed_val, population):
    """Simulate one parameter set and return cleaned feature arrays for T-cells and cancer cells."""
    num_immune, num_cancer = population
    horizon = FRAME_CUTOFF + 1  # frames 0..1800

    # Staggered observation windows sampled from the real tracks of this regime
    starts, ends = sample_population_windows(mode, num_immune, num_cancer, horizon, seed_val)

    sim = CellSimulation(
        num_immune=num_immune,
        num_cancer=num_cancer,
        width=1536.0,
        height=1536.0,
        timesteps=horizon,
        mode=mode,
        seed=seed_val,
        params_dict=p,
        track_start_frames=starts,
        track_end_frames=ends,
    )

    with torch.inference_mode():
        for t in range(horizon):
            sim.step(t)

    is_immune_mask = sim.is_immune.cpu().numpy()
    alive = sim.recorded_active_mask.cpu().numpy()
    observed = sim.recorded_observed_mask.cpu().numpy()

    # Whole-track duration, like the experimental filter: a track runs from its
    # start frame until its sampled end frame, or until the cell dies. Tracks of
    # cells still alive at the end of the horizon keep their sampled end frame.
    died = ~alive[:, -1]
    death_frame = np.where(died, np.argmax(~alive, axis=1), np.iinfo(np.int64).max)
    last_tracked = np.minimum(ends, death_frame - 1)
    track_seconds = (last_tracked - starts) * SECONDS_PER_FRAME
    has_rows = observed.any(axis=1)

    sim_tcell_qual = is_immune_mask & has_rows & (track_seconds >= IMMUNE_MIN_DT_ACC_SECONDS)
    sim_cancer_qual = ~is_immune_mask & has_rows & (track_seconds >= CANCER_MIN_DT_ACC_SECONDS)

    # Same observation grid as the experiment: every 6th frame
    frames = np.arange(horizon)
    row_mask = observed & ((frames % OBSERVATION_STRIDE == 0) & (frames <= FRAME_CUTOFF))[None, :]

    feature_tensor_map = {
        "POSITION_X": sim.rec_pos_x,
        "POSITION_Y": sim.rec_pos_y,
        "DX_FROM_PREVIOUS_POINT": sim.rec_dx_prev,
        "DY_FROM_PREVIOUS_POINT": sim.rec_dy_prev,
        "DISPLACEMENT_FROM_PREVIOUS_POINT": sim.rec_disp_prev,
        "DX_FROM_ORIGIN": sim.rec_dx_orig,
        "DY_FROM_ORIGIN": sim.rec_dy_orig,
        "DISPLACEMENT_FROM_ORIGIN": sim.rec_disp_orig,
        "DX_ACC": sim.rec_dx_acc,
        "DY_ACC": sim.rec_dy_acc,
        "DT_ACC": sim.rec_dt_acc,
        "DISTANCE_TRAVELED": sim.rec_dist_traveled,
        "PATH_EFFICIENCY": sim.rec_path_efficiency,
        "VEL_X": sim.rec_vel_x,
        "VEL_Y": sim.rec_vel_y,
        "SPEED": sim.rec_speed,
        "AVERAGE_SPEED": sim.rec_avg_speed,
    }

    def extract_clean_arrays(qualifying_cell_mask):
        mask = row_mask & qualifying_cell_mask[:, None]
        return {feat: clean_values(feat, buf.cpu().numpy()[mask]) for feat, buf in feature_tensor_map.items()}

    sim_tcell_data = extract_clean_arrays(sim_tcell_qual)
    sim_cancer_data = extract_clean_arrays(sim_cancer_qual)

    del sim, feature_tensor_map
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return sim_tcell_data, sim_cancer_data


# =============================================================================
# 6. EVALUATION ENGINE (ISOTROPIC REDUCED SCORING)
# =============================================================================
def evaluate_parameter_set(p, mode, exp_tcell, exp_cancer, population, seed_val):
    sim_tcell, sim_cancer = run_simulation_instance(p, mode, seed_val, population)

    feature_metrics = []
    total_weighted_loss = 0.0

    for feat in FEATURES_17:
        ks_t, w1_raw_t, w1_std_t, jsd_t = compute_metrics_for_feature(exp_tcell[feat], sim_tcell[feat])
        ks_c, w1_raw_c, w1_std_c, jsd_c = compute_metrics_for_feature(exp_cancer[feat], sim_cancer[feat])

        # 40% KS + 40% Standardized Wasserstein + 20% JSD
        score_t = 0.40 * ks_t + 0.40 * np.clip(w1_std_t, 0.0, 5.0) / 5.0 + 0.20 * jsd_t
        score_c = 0.40 * ks_c + 0.40 * np.clip(w1_std_c, 0.0, 5.0) / 5.0 + 0.20 * jsd_c
        feat_score = 0.50 * score_t + 0.50 * score_c

        weight = OPTIMIZED_FEATURE_WEIGHTS[feat]
        contribution = weight * feat_score
        total_weighted_loss += contribution

        feature_metrics.append({
            "Feature": feat,
            "KS_Stat_Avg": 0.5 * (ks_t + ks_c),
            "Wasserstein_Raw_Avg": 0.5 * (w1_raw_t + w1_raw_c),
            "Wasserstein_Std_Avg": 0.5 * (w1_std_t + w1_std_c),
            "JSD_Avg": 0.5 * (jsd_t + jsd_c),
            "Feature_Score": feat_score,
            "Feature_Weight": weight,
            "Weighted_Loss": contribution,
        })

    return total_weighted_loss, pd.DataFrame(feature_metrics)


def evaluate_over_seeds(p, mode, exp_tcell, exp_cancer, population, seeds):
    """Mean loss and mean per-feature metrics over several random seeds."""
    losses, dfs = [], []
    for seed_val in seeds:
        loss, df = evaluate_parameter_set(p, mode, exp_tcell, exp_cancer, population, seed_val)
        losses.append(loss)
        dfs.append(df)
    mean_df = pd.concat(dfs).groupby("Feature", sort=False).mean().reset_index()
    return float(np.mean(losses)), float(np.std(losses)), mean_df


# =============================================================================
# 7. OPTUNA SEARCH SPACE
# =============================================================================
def sample_trial_parameters(trial):
    p_scout = trial.suggest_float("scout_prob", 0.10, 0.45)
    p_msg = trial.suggest_float("messenger_prob", 0.10, 0.45)
    p_killer = max(0.10, 1.0 - (p_scout + p_msg))

    params = {
        "scout_prob": p_scout,
        "messenger_prob": p_msg,
        "killer_prob": p_killer,

        "IMMUNE_BASE_MEAN": trial.suggest_float("IMMUNE_BASE_MEAN", 3.0, 9.0),
        "IMMUNE_BASE_STD": trial.suggest_float("IMMUNE_BASE_STD", 0.8, 2.5),
        "speed_mult_scout": trial.suggest_float("speed_mult_scout", 1.05, 1.60),
        "speed_mult_msg": trial.suggest_float("speed_mult_msg", 0.85, 1.35),
        "speed_mult_killer": trial.suggest_float("speed_mult_killer", 0.55, 1.05),
        "can_sessile_min": trial.suggest_float("can_sessile_min", 0.10, 0.80),
        "can_sessile_max": trial.suggest_float("can_sessile_max", 0.80, 2.20),
        "can_evasive_min": trial.suggest_float("can_evasive_min", 1.20, 2.80),
        "can_evasive_max": trial.suggest_float("can_evasive_max", 2.80, 5.00),

        "tau_scout_min": trial.suggest_float("tau_scout_min", 0.8, 2.0),
        "tau_scout_max": trial.suggest_float("tau_scout_max", 2.0, 3.8),
        "tau_msg_min": trial.suggest_float("tau_msg_min", 1.8, 3.2),
        "tau_msg_max": trial.suggest_float("tau_msg_max", 3.2, 5.0),
        "tau_killer_min": trial.suggest_float("tau_killer_min", 2.8, 4.2),
        "tau_killer_max": trial.suggest_float("tau_killer_max", 4.2, 6.8),
        "tau_cancer_min": trial.suggest_float("tau_cancer_min", 1.2, 2.8),
        "tau_cancer_max": trial.suggest_float("tau_cancer_max", 2.8, 5.8),

        "noise_scout_min": trial.suggest_float("noise_scout_min", 0.15, 0.35),
        "noise_scout_max": trial.suggest_float("noise_scout_max", 0.35, 0.65),
        "noise_msg_min": trial.suggest_float("noise_msg_min", 0.08, 0.22),
        "noise_msg_max": trial.suggest_float("noise_msg_max", 0.22, 0.45),
        "noise_killer_min": trial.suggest_float("noise_killer_min", 0.01, 0.06),
        "noise_killer_max": trial.suggest_float("noise_killer_max", 0.06, 0.18),
        "noise_can_sess_min": trial.suggest_float("noise_can_sess_min", 0.01, 0.06),
        "noise_can_sess_max": trial.suggest_float("noise_can_sess_max", 0.06, 0.15),
        "noise_can_evas_min": trial.suggest_float("noise_can_evas_min", 0.08, 0.22),
        "noise_can_evas_max": trial.suggest_float("noise_can_evas_max", 0.22, 0.45),

        "sensing_scout_min": trial.suggest_float("sensing_scout_min", 15.0, 24.0),
        "sensing_scout_max": trial.suggest_float("sensing_scout_max", 24.0, 36.0),
        "sensing_msg_min": trial.suggest_float("sensing_msg_min", 12.0, 18.0),
        "sensing_msg_max": trial.suggest_float("sensing_msg_max", 18.0, 25.0),
        "sensing_killer_min": trial.suggest_float("sensing_killer_min", 6.0, 11.0),
        "sensing_killer_max": trial.suggest_float("sensing_killer_max", 11.0, 18.0),
        "sensing_can_min": trial.suggest_float("sensing_can_min", 9.0, 14.0),
        "sensing_can_max": trial.suggest_float("sensing_can_max", 14.0, 20.0),

        "ENERGY_DRAIN_MOVE": trial.suggest_float("ENERGY_DRAIN_MOVE", 0.0005, 0.004, log=True),
        "ENERGY_DRAIN_CHASE": trial.suggest_float("ENERGY_DRAIN_CHASE", 0.003, 0.018, log=True),
        "ENERGY_DRAIN_COMBAT": trial.suggest_float("ENERGY_DRAIN_COMBAT", 0.008, 0.035),
        "ENERGY_RECOVER_REST": trial.suggest_float("ENERGY_RECOVER_REST", 0.008, 0.040),

        "SIGNAL_EMISSION_STRENGTH": trial.suggest_float("SIGNAL_EMISSION_STRENGTH", 0.4, 1.4),
        "SIGNAL_DECAY_RATE": trial.suggest_float("SIGNAL_DECAY_RATE", 0.02, 0.12),
        "SIGNAL_SENSING_RADIUS": trial.suggest_float("SIGNAL_SENSING_RADIUS", 18.0, 40.0),
        "KILL_RADIUS": trial.suggest_float("KILL_RADIUS", 2.0, 4.2),
        "MIN_SEPARATION": trial.suggest_float("MIN_SEPARATION", 1.2, 2.5),
        "MEMORY_GAIN_FACTOR": trial.suggest_float("MEMORY_GAIN_FACTOR", 0.08, 0.35),
        "MEMORY_MAX_BONUS": trial.suggest_float("MEMORY_MAX_BONUS", 0.15, 0.40),
    }
    return params


# =============================================================================
# 8. REGIME CALIBRATION RUNNER
# =============================================================================
def run_optimization_for_regime(mode_label, run_dir, num_trials=NUM_TRIALS):
    print("\n" + "#" * 85)
    print(f"STARTING BAYESIAN CALIBRATION: {mode_label.upper()} REGIME")
    print(f"Horizon: 0 to 5 Hours (Frames 0 to {FRAME_CUTOFF}) | Balanced Objective Applied")
    print("#" * 85)

    exp_tcell = load_and_validate_experimental_data(pp.EXPERIMENTAL_DATA[mode_label]["tcell"], is_cancer=False)
    exp_cancer = load_and_validate_experimental_data(pp.EXPERIMENTAL_DATA[mode_label]["cancer"], is_cancer=True)

    population = (
        estimate_population(mode_label, "tcell", FRAME_CUTOFF, DENSITY_FRACTION),
        estimate_population(mode_label, "cancer", FRAME_CUTOFF, DENSITY_FRACTION),
    )
    print(f"Simulated population ({DENSITY_FRACTION:.0%} of real density): "
          f"{population[0]} T-cells + {population[1]} cancer cells")
    print(f"Train seeds: {list(TRAIN_SEEDS)} | Validation seeds: {list(VALIDATION_SEEDS)} "
          f"(top {TOP_K_VALIDATE} trials)\n")

    def objective(trial):
        params = sample_trial_parameters(trial)
        trial.set_user_attr("params", params)
        loss, loss_std, _ = evaluate_over_seeds(params, mode_label, exp_tcell, exp_cancer, population, TRAIN_SEEDS)
        trial.set_user_attr("train_loss_std", loss_std)
        return loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED)
    )
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)

    # Re-check the best trials on fresh seeds and keep the one that generalises best
    finished = [t for t in study.trials if t.value is not None]
    top_trials = sorted(finished, key=lambda t: t.value)[:TOP_K_VALIDATE]
    print(f"\nValidating the top {len(top_trials)} trials on fresh seeds {list(VALIDATION_SEEDS)}...")
    validation = {}
    for t in top_trials:
        v_loss, v_std, v_df = evaluate_over_seeds(t.user_attrs["params"], mode_label, exp_tcell, exp_cancer,
                                                  population, VALIDATION_SEEDS)
        validation[t.number] = (v_loss, v_std, v_df)
        print(f"  Trial #{t.number:<4} train loss = {t.value:.4f} | validation loss = {v_loss:.4f} ± {v_std:.4f}")

    best_number = min(validation, key=lambda n: validation[n][0])
    best_trial = study.trials[best_number]
    best_params = dict(best_trial.user_attrs["params"])
    best_val_loss, best_val_std, best_df = validation[best_number]

    # Export regime artifacts
    out_json = os.path.join(run_dir, f"optimal_parameters_{mode_label}_0to5hrs.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
    os.makedirs(pp.TUNING_DIR, exist_ok=True)
    shutil.copyfile(out_json, pp.optimal_params_path(mode_label))
    print(f"\n  [Artifact 1] Exported: {out_json}")
    print(f"               Current parameters updated: {pp.optimal_params_path(mode_label)}")

    trials_summary = []
    for t in finished:
        record = {"trial_number": t.number, "train_loss": t.value,
                  "train_loss_std": t.user_attrs.get("train_loss_std"),
                  "validation_loss": validation.get(t.number, (None,))[0],
                  "selected": t.number == best_number}
        record.update(t.user_attrs["params"])
        trials_summary.append(record)
    out_trials = os.path.join(run_dir, f"optimization_trials_{mode_label}_0to5hrs.csv")
    pd.DataFrame(trials_summary).to_csv(out_trials, index=False)
    print(f"  [Artifact 2] Exported: {out_trials}")

    out_metrics = os.path.join(run_dir, f"best_feature_scores_{mode_label}_0to5hrs.csv")
    best_df.to_csv(out_metrics, index=False)
    print(f"  [Artifact 3] Exported: {out_metrics}")

    print("\n" + "-" * 85)
    print(f"CALIBRATION SUMMARY — {mode_label.upper()} REGIME")
    print(f"Selected Trial #{best_number}: train loss {best_trial.value:.4f}, "
          f"validation loss {best_val_loss:.4f} ± {best_val_std:.4f}")
    print("-" * 85)
    key_targets = best_df[best_df["Feature_Weight"] > 0]
    for _, row in key_targets.iterrows():
        print(f"  Target: {row['Feature']:<32} | Score = {row['Feature_Score']:.4f} "
              f"(KS = {row['KS_Stat_Avg']:.3f}, Std_W1 = {row['Wasserstein_Std_Avg']:.3f}, JSD = {row['JSD_Avg']:.3f})")
    print("-" * 85 + "\n")

    return {"selected_trial": best_number, "train_loss": best_trial.value,
            "validation_loss": best_val_loss, "validation_loss_std": best_val_std,
            "population": {"tcell": population[0], "cancer": population[1]}}


# =============================================================================
# 9. MAIN EXECUTION (SEQUENTIAL RUN ACROSS BOTH REGIMES)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Tune simulator parameters against the experimental kinematics.")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS, help="Optuna trials per regime")
    parser.add_argument("--regimes", nargs="+", default=list(pp.REGIMES), choices=list(pp.REGIMES))
    args = parser.parse_args()

    run_dir = pp.new_run_dir(pp.TUNING_DIR)

    print("=" * 85)
    print("BALANCED DUAL-REGIME CALIBRATION PIPELINE")
    print("Active Targets: DX_PREV, DISP_PREV, SPEED, DX_ORIG, DISP_ORIG, DX_ACC, DIST_TRAV, PATH_EFF")
    print(f"Output directory: {run_dir}")
    print("=" * 85)

    run_config = {
        "trials_per_regime": args.trials, "frame_cutoff": FRAME_CUTOFF,
        "observation_stride": OBSERVATION_STRIDE, "density_fraction": DENSITY_FRACTION,
        "train_seeds": list(TRAIN_SEEDS), "validation_seeds": list(VALIDATION_SEEDS),
        "top_k_validate": TOP_K_VALIDATE, "sampler_seed": SAMPLER_SEED,
        "feature_weights": OPTIMIZED_FEATURE_WEIGHTS, "regimes": {},
    }
    for regime in args.regimes:
        run_config["regimes"][regime] = run_optimization_for_regime(regime, run_dir, args.trials)
        with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=4)

    print("=" * 85)
    print("OPTIMIZATION COMPLETE")
    print(f"Artifacts: {run_dir}")
    print(f"Current optimal parameters: {pp.TUNING_DIR}")
    print("=" * 85)


if __name__ == "__main__":
    main()
