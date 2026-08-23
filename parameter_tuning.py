"""
parameter_tuning.py
================================================================================
Visual & Concise Numerical Parameter-Sweep Pipeline for Cell Kinematics
--------------------------------------------------------------------------------
1. Executes GPU simulation across parameter sweeps (supports scalars & tuples).
2. Generates 4 14-feature raw physical histograms + 1 summary overview per run.
3. Exports a concise 10-metric CSV (raw_metrics_summary.csv) per parameter setting:
     - Exp Mean, Sim Mean, Abs Mean Diff
     - Exp Median, Sim Median, Abs Median Diff
     - Exp Std, Sim Std, Abs Std Diff
     - Raw Wasserstein Distance, Two-Sample KS Statistic
================================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. USER CONFIGURATION (SET YOUR PARAMETERS HERE)
# =============================================================================

# Name of the simulator parameter to tune:
# Examples: "tau", "noise_scale", "IMMUNE_BASE_MEAN", "CAN_EVASIVE_SPEED", "max_speed"
TUNING_PARAMETER = "KILL_RADIUS"

# Range tuples [(min, max), ...] OR scalar values [val1, val2, ...]
TUNING_VALUES = [2.5]

# Random seed for fair stochastic comparison across parameter iterations
RANDOM_SEED = 42

# Root output directory
BASE_OUTPUT_DIR = r"parameter_tuning"


# =============================================================================
# 2. EXPERIMENTAL DATASETS & 14 KINEMATIC FEATURES
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

NON_NEGATIVE_FEATURES = [
    "DISPLACEMENT_FROM_PREVIOUS_POINT",
    "DISPLACEMENT_FROM_ORIGIN",
    "DISTANCE_TRAVELED",
    "PATH_EFFICIENCY",
    "SPEED",
    "AVERAGE_SPEED",
]


# =============================================================================
# 3. CELL SIMULATOR ENGINE (PyTorch GPU)
# =============================================================================
_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),  (0, 0),  (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]


def _sample_uniform(lo, hi, n, generator, device):
    return lo + (hi - lo) * torch.rand(n, generator=generator, device=device)


class CellSimulation:
    def __init__(self,
                 num_immune=100, num_cancer=100,
                 width=1536.0, height=1536.0,
                 timesteps=8635,
                 mode='killing',
                 dt=1.0,
                 seed=None,
                 target_density=0.002,
                 max_per_cell=16,
                 max_signals_per_cell=32,
                 device=None,
                 param_override=None):

        self.device = device if device is not None else DEVICE
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))
        else:
            self.gen.seed()

        self.num_immune = num_immune
        self.num_cancer = num_cancer
        self.num_cells = N = num_immune + num_cancer
        self.timesteps = timesteps
        self.dt = dt
        self.mode = mode
        self.max_per_cell = max_per_cell
        self.max_signals_per_cell = max_signals_per_cell
        self.param_override = param_override if param_override is not None else {}

        self.width = float(width)
        self.height = float(height)

        self.scout_prob = 0.30
        self.messenger_prob = 0.30
        self.killer_prob = 0.40
        self.target_lock_timeout = 15

        self._initialize_constants()
        self._initialize_phenotypes()
        self._initialize_motion_parameters()
        self._initialize_signals()
        self._initialize_dynamic_state()

        self._accel = torch.zeros((self.num_cells, 2), device=self.device)

    def _blend(self, mask, lo, hi, base):
        return torch.where(mask, _sample_uniform(lo, hi, self.num_cells, self.gen, self.device), base)

    def _initialize_constants(self):
        if self.mode == 'killing':
            self.IMMUNE_BASE_MEAN, self.IMMUNE_BASE_STD = 5.0, 1.6
            self.RECOG_BASE = (0.85, 0.60, 0.95)
            self.SIGNAL_EMISSION_STRENGTH = 1.00
        else:
            self.IMMUNE_BASE_MEAN, self.IMMUNE_BASE_STD = 8.5, 2.0
            self.RECOG_BASE = (0.25, 0.15, 0.30)
            self.SIGNAL_EMISSION_STRENGTH = 0.35

        self.SPEED_MULTS = (1.35, 1.00, 0.75)
        self.CAN_SESSILE_SPEED = (0.5, 1.5)
        self.CAN_EVASIVE_SPEED = (2.0, 4.0)

        self.SENSING_SCOUT = (24.0, 32.0)
        self.SENSING_MESSENGER = (16.0, 22.0)
        self.SENSING_KILLER = (8.0, 14.0)
        self.SENSING_CANCER = (12.0, 18.0)
        self.EVASIVE_HIDE_PENALTY = 0.40

        self.MAX_SIGNALS = 2000
        self.SIGNAL_DECAY_RATE = 0.05
        self.SIGNAL_SENSING_RADIUS = 30.0
        self.SIGNAL_MAX_CAP = 3.0
        self.AMPLIFY_COOLDOWN_STEPS = 5
        self.CHEMOTAXIS_EPSILON = 2.0

        self.SIGNAL_LIFETIME_SCOUT_MEAN = 25.0
        self.SIGNAL_LIFETIME_SCOUT_STD = 5.0
        self.SIGNAL_LIFETIME_MSG_MEAN = 15.0
        self.SIGNAL_LIFETIME_MSG_STD = 3.0

        self.KILL_RADIUS = 2.5
        self.ENGAGE_STEPS_REQUIRED = 4
        self.MIN_SEPARATION = 1.5
        self.BOUNDARY_MARGIN = 8.0
        self.KILL_RATES = (0.01, 0.00, 0.35 if self.mode == 'killing' else 0.0175)

        self.MEMORY_DECAY_RATE = 0.005
        self.MEMORY_GAIN_FACTOR = 0.30
        self.MEMORY_MAX_BONUS = 0.30

        self.ENERGY_DRAIN_MOVE = 0.002
        self.ENERGY_DRAIN_CHASE = 0.005
        self.ENERGY_DRAIN_COMBAT = 0.020
        self.ENERGY_RECOVER_REST = 0.015
        self.ENERGY_EMIT_THRESHOLD = 0.15

        self.ALIGNMENT_RADIUS_KILLER = 12.0
        self.ALIGNMENT_RADIUS_CANCER = 10.0

        for k, v in self.param_override.items():
            if hasattr(self, k) and not isinstance(v, tuple):
                setattr(self, k, float(v))

    def _initialize_phenotypes(self):
        dev = self.device
        gen = self.gen

        self.base_class = torch.cat([
            torch.ones(self.num_immune, dtype=torch.long, device=dev),
            2 * torch.ones(self.num_cancer, dtype=torch.long, device=dev),
        ])
        self.is_immune = self.base_class == 1
        self.is_cancer = self.base_class == 2

        p_rand = torch.rand(self.num_immune, generator=gen, device=dev)
        t_scout = self.scout_prob
        t_msg = self.scout_prob + self.messenger_prob

        phen_immune = torch.where(p_rand < t_scout, torch.tensor(0, device=dev),
                      torch.where(p_rand < t_msg, torch.tensor(1, device=dev),
                                                  torch.tensor(2, device=dev)))
        phen_cancer = (torch.rand(self.num_cancer, generator=gen, device=dev) < 0.4).long()
        self.phenotype = torch.cat([phen_immune, phen_cancer])

    def _initialize_motion_parameters(self):
        dev = self.device
        gen = self.gen
        N = self.num_cells

        imm_scout = self.is_immune & (self.phenotype == 0)
        imm_msg = self.is_immune & (self.phenotype == 1)
        imm_killer = self.is_immune & (self.phenotype == 2)
        can_sessile = self.is_cancer & (self.phenotype == 0)
        can_evasive = self.is_cancer & (self.phenotype == 1)

        raw_speeds = torch.normal(mean=self.IMMUNE_BASE_MEAN, std=self.IMMUNE_BASE_STD,
                                  size=(N,), generator=gen, device=dev)
        neg_mask = raw_speeds <= 0.0
        while neg_mask.any():
            num_bad = int(neg_mask.sum().item())
            replacements = torch.normal(mean=self.IMMUNE_BASE_MEAN, std=self.IMMUNE_BASE_STD,
                                        size=(num_bad,), generator=gen, device=dev)
            raw_speeds[neg_mask] = replacements
            neg_mask = raw_speeds <= 0.0

        z = torch.zeros(N, device=dev)
        self.max_speed = torch.where(imm_scout, raw_speeds * self.SPEED_MULTS[0], z)
        self.max_speed = torch.where(imm_msg, raw_speeds * self.SPEED_MULTS[1], self.max_speed)
        self.max_speed = torch.where(imm_killer, raw_speeds * self.SPEED_MULTS[2], self.max_speed)
        self.max_speed = self._blend(can_sessile, self.CAN_SESSILE_SPEED[0], self.CAN_SESSILE_SPEED[1], self.max_speed)
        self.max_speed = self._blend(can_evasive, self.CAN_EVASIVE_SPEED[0], self.CAN_EVASIVE_SPEED[1], self.max_speed)

        self.sensing_radius = self._blend(imm_scout, self.SENSING_SCOUT[0], self.SENSING_SCOUT[1], z.clone())
        self.sensing_radius = self._blend(imm_msg, self.SENSING_MESSENGER[0], self.SENSING_MESSENGER[1], self.sensing_radius)
        self.sensing_radius = self._blend(imm_killer, self.SENSING_KILLER[0], self.SENSING_KILLER[1], self.sensing_radius)
        self.sensing_radius = torch.where(self.is_cancer,
                                           _sample_uniform(self.SENSING_CANCER[0], self.SENSING_CANCER[1], N, gen, dev),
                                           self.sensing_radius)

        recog_mean = torch.where(imm_scout, self.RECOG_BASE[0],
                     torch.where(imm_msg, self.RECOG_BASE[1],
                     torch.where(imm_killer, self.RECOG_BASE[2], 1.0)))
        
        recog_samples = torch.normal(mean=recog_mean, std=0.08, generator=gen)
        self.base_recognition_prob = torch.clamp(recog_samples, 0.05, 1.0)

        self.kill_rate = torch.where(imm_scout, self.KILL_RATES[0],
                         torch.where(imm_msg, self.KILL_RATES[1],
                         torch.where(imm_killer, self.KILL_RATES[2], 0.0)))

        # Handle 'tau' parameter override or default assignment
        if "tau" in self.param_override:
            tau_val = self.param_override["tau"]
            if isinstance(tau_val, tuple):
                t_lo, t_hi = tau_val
                self.tau = _sample_uniform(t_lo, t_hi, N, gen, dev)
            else:
                self.tau = torch.full((N,), float(tau_val), device=dev)
        else:
            self.tau = self._blend(imm_scout, 1.0, 2.0, z.clone())
            self.tau = self._blend(imm_msg, 2.0, 3.5, self.tau)
            self.tau = self._blend(imm_killer, 3.5, 5.0, self.tau)
            self.tau = torch.where(self.is_cancer, _sample_uniform(2.0, 4.0, N, gen, dev), self.tau)

        # Handle 'noise_scale' parameter override or default assignment
        if "noise_scale" in self.param_override:
            ns_val = self.param_override["noise_scale"]
            if isinstance(ns_val, tuple):
                ns_lo, ns_hi = ns_val
                self.noise_scale = _sample_uniform(ns_lo, ns_hi, N, gen, dev)
            else:
                self.noise_scale = torch.full((N,), float(ns_val), device=dev)
        else:
            self.noise_scale = self._blend(imm_scout, 0.20, 0.30, z.clone())
            self.noise_scale = self._blend(imm_msg, 0.10, 0.20, self.noise_scale)
            self.noise_scale = self._blend(imm_killer, 0.01, 0.05, self.noise_scale)
            self.noise_scale = self._blend(can_sessile, 0.02, 0.06, self.noise_scale)
            self.noise_scale = self._blend(can_evasive, 0.15, 0.35, self.noise_scale)

    def _initialize_signals(self):
        dev = self.device
        S = self.MAX_SIGNALS

        self.signal_pos = torch.zeros((S, 2), device=dev)
        self.signal_strength = torch.zeros(S, device=dev)
        self.signal_cancer_id = torch.full((S,), -1, dtype=torch.long, device=dev)
        self.signal_emitter_id = torch.full((S,), -1, dtype=torch.long, device=dev)
        self.signal_emitter_type = torch.full((S,), -1, dtype=torch.long, device=dev)
        self.signal_timestamp = torch.zeros(S, dtype=torch.long, device=dev)
        self.signal_lifetime = torch.zeros(S, dtype=torch.long, device=dev)
        self.signal_active = torch.zeros(S, dtype=torch.bool, device=dev)

    def _initialize_dynamic_state(self):
        dev = self.device
        gen = self.gen
        N = self.num_cells
        T = self.timesteps

        immune_ids = torch.arange(1, self.num_immune + 1, dtype=torch.long, device=dev)
        cancer_ids = torch.arange(101, 101 + self.num_cancer, dtype=torch.long, device=dev)
        self.track_id = torch.cat([immune_ids, cancer_ids])

        self.positions = torch.stack([
            _sample_uniform(0, self.width, N, gen, dev),
            _sample_uniform(0, self.height, N, gen, dev),
        ], dim=1)

        init_speeds = torch.rand(N, generator=gen, device=dev) * self.max_speed
        init_angles = torch.rand(N, generator=gen, device=dev) * (2.0 * math.pi)
        self.velocities = torch.stack([
            init_speeds * torch.cos(init_angles),
            init_speeds * torch.sin(init_angles)
        ], dim=1)

        self.active_mask = torch.ones(N, dtype=torch.bool, device=dev)
        self.energy = torch.ones(N, device=dev)
        self.contact_timer = torch.zeros(N, device=dev)
        self.immune_contact_timer = torch.zeros(N, device=dev)
        self.locked_target = torch.full((N,), -1, dtype=torch.long, device=dev)

        self.memory_bonus = torch.zeros(N, device=dev)
        self.combat_experience = torch.zeros(N, device=dev)
        self.last_cancer_seen_time = torch.zeros(N, dtype=torch.long, device=dev)
        self.target_lost_timer = torch.zeros(N, dtype=torch.long, device=dev)
        self.messenger_amp_cooldown = torch.zeros(N, dtype=torch.long, device=dev)

        widest_radius = max(float((self.sensing_radius * 1.5).max().item()), self.SIGNAL_SENSING_RADIUS)
        self.cell_size = max(widest_radius, self.MIN_SEPARATION) + 1e-3
        self.grid_w = max(1, int(math.ceil(self.width / self.cell_size)))
        self.grid_h = max(1, int(math.ceil(self.height / self.cell_size)))
        self.num_grid_cells = self.grid_w * self.grid_h

        self.recorded_active_mask = torch.zeros((N, T), dtype=torch.bool, device=dev)
        self.rec_pos_x = torch.zeros((N, T), device=dev)
        self.rec_pos_y = torch.zeros((N, T), device=dev)
        self.rec_dx_prev = torch.full((N, T), float('nan'), device=dev)
        self.rec_dy_prev = torch.full((N, T), float('nan'), device=dev)
        self.rec_disp_prev = torch.full((N, T), float('nan'), device=dev)
        self.rec_dx_orig = torch.zeros((N, T), device=dev)
        self.rec_dy_orig = torch.zeros((N, T), device=dev)
        self.rec_disp_orig = torch.zeros((N, T), device=dev)
        self.rec_dist_traveled = torch.zeros((N, T), device=dev)
        self.rec_path_efficiency = torch.zeros((N, T), device=dev)
        self.rec_vel_x = torch.full((N, T), float('nan'), device=dev)
        self.rec_vel_y = torch.full((N, T), float('nan'), device=dev)
        self.rec_speed = torch.full((N, T), float('nan'), device=dev)
        self.rec_avg_speed = torch.zeros((N, T), device=dev)

        self.initial_pos = self.positions.clone()
        self.cum_dist_traveled = torch.zeros(N, device=dev)

    def build_spatial_grid(self):
        gx = torch.clamp((self.positions[:, 0] / self.cell_size).long(), 0, self.grid_w - 1)
        gy = torch.clamp((self.positions[:, 1] / self.cell_size).long(), 0, self.grid_h - 1)
        cell_id = gy * self.grid_w + gx
        cell_id = torch.where(self.active_mask, cell_id, self.num_grid_cells)

        order = torch.argsort(cell_id)
        sorted_cell_id = cell_id[order]
        counts = torch.bincount(sorted_cell_id, minlength=self.num_grid_cells + 1)
        cell_start = torch.cumsum(counts, dim=0) - counts
        arange_n = torch.arange(self.num_cells, device=self.device)
        rank_in_cell = arange_n - cell_start[sorted_cell_id]
        slot = torch.clamp(rank_in_cell, max=self.max_per_cell - 1)

        table = torch.full((self.num_grid_cells + 1, self.max_per_cell), -1,
                            dtype=torch.long, device=self.device)
        table[sorted_cell_id, slot] = order
        return table, gx, gy

    def build_signal_spatial_grid(self):
        gx = torch.clamp((self.signal_pos[:, 0] / self.cell_size).long(), 0, self.grid_w - 1)
        gy = torch.clamp((self.signal_pos[:, 1] / self.cell_size).long(), 0, self.grid_h - 1)
        cell_id = gy * self.grid_w + gx
        cell_id = torch.where(self.signal_active, cell_id, self.num_grid_cells)

        order = torch.argsort(cell_id)
        sorted_cell_id = cell_id[order]
        counts = torch.bincount(sorted_cell_id, minlength=self.num_grid_cells + 1)
        cell_start = torch.cumsum(counts, dim=0) - counts
        arange_s = torch.arange(self.MAX_SIGNALS, device=self.device)
        rank_in_cell = arange_s - cell_start[sorted_cell_id]
        slot = torch.clamp(rank_in_cell, max=self.max_signals_per_cell - 1)

        sig_table = torch.full((self.num_grid_cells + 1, self.max_signals_per_cell), -1,
                               dtype=torch.long, device=self.device)
        sig_table[sorted_cell_id, slot] = order
        return sig_table

    def find_neighbors(self, cached_gx=None, cached_gy=None):
        if cached_gx is None or cached_gy is None:
            table, gx, gy = self.build_spatial_grid()
        else:
            table, _, _ = self.build_spatial_grid()
            gx, gy = cached_gx, cached_gy

        chunks = []
        for dx, dy in _NEIGHBOR_OFFSETS:
            ngx = torch.clamp(gx + dx, 0, self.grid_w - 1)
            ngy = torch.clamp(gy + dy, 0, self.grid_h - 1)
            ncell = ngy * self.grid_w + ngx
            chunks.append(table[ncell])
        candidates = torch.cat(chunks, dim=1)
        valid = candidates >= 0

        cand_pos = self.positions[candidates.clamp(min=0)]
        diff = cand_pos - self.positions.unsqueeze(1)
        dist = torch.norm(diff, dim=2)

        self_mask = candidates == torch.arange(self.num_cells, device=self.device).unsqueeze(1)
        invalid = (~valid) | self_mask | (~self.active_mask[candidates.clamp(min=0)])
        dist = dist.masked_fill(invalid, float('inf'))

        return candidates, valid, dist, diff

    def find_nearby_signals(self, gx, gy):
        sig_table = self.build_signal_spatial_grid()
        chunks = []
        for dx, dy in _NEIGHBOR_OFFSETS:
            ngx = torch.clamp(gx + dx, 0, self.grid_w - 1)
            ngy = torch.clamp(gy + dy, 0, self.grid_h - 1)
            ncell = ngy * self.grid_w + ngx
            chunks.append(sig_table[ncell])
        sig_candidates = torch.cat(chunks, dim=1)
        valid = sig_candidates >= 0

        sig_cand_pos = self.signal_pos[sig_candidates.clamp(min=0)]
        diff = sig_cand_pos - self.positions.unsqueeze(1)
        dist = torch.norm(diff, dim=2)

        invalid = (~valid) | (~self.signal_active[sig_candidates.clamp(min=0)])
        dist = dist.masked_fill(invalid, float('inf'))

        return sig_candidates, dist, diff

    def emit_signals(self, emit_mask, positions, strengths, cancer_ids, emitter_ids, emitter_types, t):
        if not emit_mask.any():
            return

        emit_indices = torch.nonzero(emit_mask, as_tuple=False).squeeze(1)
        num_e = emit_indices.shape[0]

        e_pos = positions[emit_indices]
        e_str = strengths[emit_indices]
        e_cid = cancer_ids[emit_indices]
        e_eid = emitter_ids[emit_indices]
        e_ety = emitter_types[emit_indices]

        is_scout = (e_ety == 0)
        e_lt = torch.where(
            is_scout,
            torch.normal(mean=self.SIGNAL_LIFETIME_SCOUT_MEAN, std=self.SIGNAL_LIFETIME_SCOUT_STD, size=(num_e,), generator=self.gen, device=self.device),
            torch.normal(mean=self.SIGNAL_LIFETIME_MSG_MEAN, std=self.SIGNAL_LIFETIME_MSG_STD, size=(num_e,), generator=self.gen, device=self.device)
        ).long().clamp(min=5)

        inactive_slots = torch.nonzero(~self.signal_active, as_tuple=False).squeeze(1)

        if inactive_slots.shape[0] >= num_e:
            target_slots = inactive_slots[:num_e]
        else:
            c_alive_map = self.active_mask[self.signal_cancer_id.clamp(min=0)]
            dead_target_scores = torch.where(~c_alive_map & self.signal_active, 100.0, 0.0)
            score = dead_target_scores + (1.0 / (self.signal_strength + 1e-3))
            sorted_slots = torch.argsort(score, descending=True)
            target_slots = sorted_slots[:num_e]

        self.signal_pos[target_slots] = e_pos
        self.signal_strength[target_slots] = torch.clamp(e_str, max=self.SIGNAL_MAX_CAP)
        self.signal_cancer_id[target_slots] = e_cid
        self.signal_emitter_id[target_slots] = e_eid
        self.signal_emitter_type[target_slots] = e_ety
        self.signal_timestamp[target_slots] = t
        self.signal_lifetime[target_slots] = e_lt
        self.signal_active[target_slots] = True

    def update_signals(self, t):
        if not self.signal_active.any():
            return

        self.signal_strength = torch.where(self.signal_active,
                                            self.signal_strength * (1.0 - self.SIGNAL_DECAY_RATE),
                                            self.signal_strength)

        cancer_alive = self.active_mask[self.signal_cancer_id.clamp(min=0)]
        target_dead = (~cancer_alive) & (self.signal_cancer_id >= 0)
        expired = (t - self.signal_timestamp) >= self.signal_lifetime
        faded = self.signal_strength < 0.02

        self.signal_active = self.signal_active & (~target_dead) & (~expired) & (~faded)

    def update_immune_cells(self, candidates, dist, diff, gx, gy, t):
        dev = self.device
        imm_scout = self.is_immune & (self.phenotype == 0) & self.active_mask
        imm_msg = self.is_immune & (self.phenotype == 1) & self.active_mask
        imm_killer = self.is_immune & (self.phenotype == 2) & self.active_mask

        cand_is_cancer = self.is_cancer[candidates.clamp(min=0)]
        cand_phenotype = self.phenotype[candidates.clamp(min=0)]
        dist_cancer = dist.clone().masked_fill(~cand_is_cancer, float('inf'))

        eff_sensing = self.sensing_radius.unsqueeze(1) * (0.5 + 0.5 * self.energy.unsqueeze(1))
        within_range = dist_cancer <= eff_sensing

        effective_recog = torch.clamp(self.base_recognition_prob + self.memory_bonus, max=1.0)
        recog_matrix = effective_recog.unsqueeze(1).expand(-1, candidates.shape[1])
        
        is_evasive_target = cand_is_cancer & (cand_phenotype == 1)
        recog_matrix = torch.where(is_evasive_target,
                                   torch.clamp(recog_matrix - self.EVASIVE_HIDE_PENALTY, min=0.05),
                                   recog_matrix)

        detect_roll = torch.rand(candidates.shape, generator=self.gen, device=dev) < recog_matrix
        valid_detect = within_range & detect_roll

        tgt = self.locked_target
        tgt_valid = (tgt >= 0) & self.active_mask[tgt.clamp(min=0)]
        tgt_positions = self.positions[tgt.clamp(min=0)]
        d_tgt = torch.norm(tgt_positions - self.positions, dim=1)
        tgt_in_sensing = tgt_valid & (d_tgt <= eff_sensing[:, 0])

        self.target_lost_timer = torch.where(tgt_valid & (~tgt_in_sensing), self.target_lost_timer + 1, torch.zeros_like(self.target_lost_timer))
        lock_expired = self.target_lost_timer > self.target_lock_timeout
        tgt = torch.where(tgt_valid & (~lock_expired), tgt, -1)

        search_dist = dist_cancer.clone().masked_fill(~valid_detect, float('inf'))
        nearest_val, nearest_idx = torch.min(search_dist, dim=1)
        has_any_within = nearest_val < float('inf')

        nearest_particle = torch.gather(candidates, 1, nearest_idx.unsqueeze(1)).squeeze(1)

        acquire = (tgt < 0) & has_any_within & self.is_immune & self.active_mask
        tgt = torch.where(acquire, nearest_particle, tgt)
        self.locked_target = tgt
        has_target = (tgt >= 0) & self.is_immune & self.active_mask

        self.last_cancer_seen_time = torch.where(has_any_within, t, self.last_cancer_seen_time)

        target_pos = self.positions[tgt.clamp(min=0)]
        direction = target_pos - self.positions
        d = torch.norm(direction, dim=1) + 1e-6
        
        desired_speed = self.max_speed * (0.2 + 0.8 * self.energy)
        desired_vel_target = direction / d.unsqueeze(1) * desired_speed.unsqueeze(1)

        scout_emitting = imm_scout & has_any_within & (self.energy > self.ENERGY_EMIT_THRESHOLD)
        scout_cancer_target = self.locked_target
        self.emit_signals(
            scout_emitting, self.positions,
            torch.full((self.num_cells,), self.SIGNAL_EMISSION_STRENGTH, device=dev),
            scout_cancer_target, torch.arange(self.num_cells, device=dev),
            torch.zeros(self.num_cells, dtype=torch.long, device=dev), t
        )

        sig_candidates, sig_dist, sig_diff = self.find_nearby_signals(gx, gy)
        sig_valid = (sig_dist <= self.SIGNAL_SENSING_RADIUS)
        sig_weights = (self.signal_strength[sig_candidates.clamp(min=0)] / (sig_dist**2 + self.CHEMOTAXIS_EPSILON)).masked_fill(~sig_valid, 0.0)
        
        total_weight = sig_weights.sum(dim=1, keepdim=True) + 1e-6
        has_signal = (sig_weights.sum(dim=1) > 0.0)

        sig_cand_pos = self.signal_pos[sig_candidates.clamp(min=0)]
        weighted_signal_dirs = ((sig_cand_pos - self.positions.unsqueeze(1)) * sig_weights.unsqueeze(2)).sum(dim=1) / total_weight
        sig_dir_norm = torch.norm(weighted_signal_dirs, dim=1, keepdim=True) + 1e-6
        desired_vel_signal = (weighted_signal_dirs / sig_dir_norm) * desired_speed.unsqueeze(1)

        max_sig_val, max_sig_idx = torch.max(sig_weights, dim=1)
        best_sig_particle = torch.gather(sig_candidates, 1, max_sig_idx.unsqueeze(1)).squeeze(1)
        best_sig_emitter_type = self.signal_emitter_type[best_sig_particle.clamp(min=0)]
        is_scout_signal = (best_sig_emitter_type == 0)

        can_amplify = (t - self.messenger_amp_cooldown) >= self.AMPLIFY_COOLDOWN_STEPS
        messenger_amp = imm_msg & has_signal & is_scout_signal & can_amplify & (self.energy > 0.2)
        if messenger_amp.any():
            self.messenger_amp_cooldown = torch.where(messenger_amp, t, self.messenger_amp_cooldown)
            target_c_ids = self.signal_cancer_id[best_sig_particle.clamp(min=0)]
            self.emit_signals(
                messenger_amp, self.positions,
                max_sig_val * 1.4, target_c_ids,
                torch.arange(self.num_cells, device=dev),
                torch.ones(self.num_cells, dtype=torch.long, device=dev), t
            )

        cand_is_killer = (self.is_immune & (self.phenotype == 2))[candidates.clamp(min=0)]
        cand_vel = self.velocities[candidates.clamp(min=0)]
        killer_align_mask = cand_is_killer.unsqueeze(2) & (dist < self.ALIGNMENT_RADIUS_KILLER).unsqueeze(2)
        avg_killer_vel = (cand_vel * killer_align_mask.float()).sum(dim=1) / (killer_align_mask.sum(dim=1).clamp(min=1))
        align_dir = avg_killer_vel / (torch.norm(avg_killer_vel, dim=1, keepdim=True) + 1e-6)

        random_explore = torch.randn(self.num_cells, 2, generator=self.gen, device=dev) * (self.max_speed * 0.3).unsqueeze(1)

        desired_vel = torch.where(has_target.unsqueeze(1), desired_vel_target,
                      torch.where(has_signal.unsqueeze(1) & (imm_msg | imm_killer).unsqueeze(1),
                                  0.7 * desired_vel_signal + 0.3 * align_dir * desired_speed.unsqueeze(1),
                                  random_explore))

        steer_accel = (desired_vel - self.velocities) / self.tau.unsqueeze(1)
        apply_mask = (self.is_immune & self.active_mask).unsqueeze(1)
        self._accel += torch.where(apply_mask, steer_accel, 0.0)

        moving = (torch.norm(self.velocities, dim=1) > 0.5) & self.is_immune & self.active_mask
        chasing = has_target & self.is_immune & self.active_mask
        resting = (~moving) & (~chasing) & self.is_immune & self.active_mask

        energy_drain = torch.where(chasing, self.ENERGY_DRAIN_CHASE,
                       torch.where(moving, self.ENERGY_DRAIN_MOVE, 0.0))
        energy_gain = torch.where(resting, self.ENERGY_RECOVER_REST, 0.0)
        self.energy = torch.clamp(self.energy - energy_drain + energy_gain, 0.05, 1.0)

        unseen_duration = (t - self.last_cancer_seen_time)
        decay_mask = self.is_immune & self.active_mask & (unseen_duration > 5)
        self.memory_bonus = torch.where(decay_mask, torch.clamp(self.memory_bonus - self.MEMORY_DECAY_RATE * self.dt, min=0.0), self.memory_bonus)
        self.combat_experience = torch.where(decay_mask, torch.clamp(self.combat_experience - self.MEMORY_DECAY_RATE * self.dt, min=0.0), self.combat_experience)

    def update_cancer_cells(self, candidates, dist, diff):
        dev = self.device
        is_evasive = self.is_cancer & (self.phenotype == 1)
        cand_is_immune = self.is_immune[candidates.clamp(min=0)]
        dist_immune = dist.clone().masked_fill(~cand_is_immune, float('inf'))

        within = dist_immune <= self.sensing_radius.unsqueeze(1)
        nearest_idx = torch.argmin(dist_immune, dim=1)
        has_threat = within.any(dim=1) & is_evasive & self.active_mask

        flee_dir = -torch.gather(diff, 1, nearest_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        
        cand_is_evasive_cancer = (self.is_cancer & (self.phenotype == 1))[candidates.clamp(min=0)]
        cand_vel = self.velocities[candidates.clamp(min=0)]
        cancer_align_mask = cand_is_evasive_cancer.unsqueeze(2) & (dist < self.ALIGNMENT_RADIUS_CANCER).unsqueeze(2)
        avg_cancer_vel = (cand_vel * cancer_align_mask.float()).sum(dim=1) / (cancer_align_mask.sum(dim=1).clamp(min=1))

        combined_flee = 0.75 * flee_dir + 0.25 * avg_cancer_vel
        dnorm = torch.norm(combined_flee, dim=1) + 1e-6
        desired_vel_flee = combined_flee / dnorm.unsqueeze(1) * self.max_speed.unsqueeze(1)
        steer_flee = (desired_vel_flee - self.velocities) / self.tau.unsqueeze(1)

        drift = torch.randn(self.num_cells, 2, generator=self.gen, device=dev) * 0.02 - 0.10 * self.velocities

        cancer_accel = torch.where(has_threat.unsqueeze(1), steer_flee, drift)
        apply_mask = (self.is_cancer & self.active_mask).unsqueeze(1)
        self._accel += torch.where(apply_mask, cancer_accel, 0.0)

    def apply_environment_noise(self):
        dev = self.device
        eff_noise = self.noise_scale * (1.5 - 0.5 * self.energy)
        noise = torch.randn(self.num_cells, 2, generator=self.gen, device=dev) * eff_noise.unsqueeze(1)
        apply_mask = self.active_mask.unsqueeze(1)
        self._accel += torch.where(apply_mask, noise, 0.0)

    def resolve_collisions(self, candidates, valid, dist, diff):
        close = valid & (dist < self.MIN_SEPARATION) & self.active_mask.unsqueeze(1) & self.active_mask[candidates.clamp(min=0)]
        overlap = torch.clamp(self.MIN_SEPARATION - dist, min=0)
        unit_away = -diff / (dist.unsqueeze(2) + 1e-6)
        repulse = (unit_away * overlap.unsqueeze(2) * close.unsqueeze(2)).sum(dim=1) * 0.5
        self._accel += repulse

    def apply_boundary_forces(self):
        x, y = self.positions[:, 0], self.positions[:, 1]
        push_x = torch.where(x < self.BOUNDARY_MARGIN, (self.BOUNDARY_MARGIN - x) * 0.3, 0.0)
        push_x = torch.where(x > self.width - self.BOUNDARY_MARGIN, -(x - (self.width - self.BOUNDARY_MARGIN)) * 0.3, push_x)
        push_y = torch.where(y < self.BOUNDARY_MARGIN, (self.BOUNDARY_MARGIN - y) * 0.3, 0.0)
        push_y = torch.where(y > self.height - self.BOUNDARY_MARGIN, -(y - (self.height - self.BOUNDARY_MARGIN)) * 0.3, push_y)
        boundary_accel = torch.stack([push_x, push_y], dim=1)
        apply_mask = self.active_mask.unsqueeze(1)
        self._accel += torch.where(apply_mask, boundary_accel, 0.0)

    def integrate_motion(self):
        active = self.active_mask.unsqueeze(1)
        self.velocities = torch.where(active, self.velocities + self._accel * self.dt, 0.0)
        speed = torch.norm(self.velocities, dim=1) + 1e-9
        scale = torch.clamp(self.max_speed / speed, max=1.0)
        self.velocities = self.velocities * scale.unsqueeze(1)
        self.positions = torch.where(active, self.positions + self.velocities * self.dt, self.positions)
        self.positions[:, 0] = torch.clamp(self.positions[:, 0], 0.0, self.width)
        self.positions[:, 1] = torch.clamp(self.positions[:, 1], 0.0, self.height)

    def perform_killing_phase1_immune_attack(self, candidates, dist, t):
        dev = self.device
        cand_is_immune = self.is_immune[candidates.clamp(min=0)]
        dist_immune = dist.clone().masked_fill(~cand_is_immune, float('inf'))
        nearest_val, nearest_idx = torch.min(dist_immune, dim=1)

        nearest_particle = torch.gather(candidates, 1, nearest_idx.unsqueeze(1)).squeeze(1)

        is_cancer_active = self.is_cancer & self.active_mask
        in_contact = is_cancer_active & (nearest_val < self.KILL_RADIUS)
        self.contact_timer = torch.where(in_contact, self.contact_timer + 1, 0.0)

        ready = in_contact & (self.contact_timer >= self.ENGAGE_STEPS_REQUIRED)
        attacker = nearest_particle.clamp(min=0)

        saturating_experience = 0.25 * (1.0 - torch.exp(-2.0 * self.combat_experience[attacker]))
        effective_kill_rate = self.kill_rate[attacker] + saturating_experience
        
        p_kill = effective_kill_rate * (0.2 + 0.8 * self.energy[attacker])
        roll = torch.rand(self.num_cells, generator=self.gen, device=dev)
        killed_now = ready & (roll < p_kill)

        self.active_mask = self.active_mask & (~killed_now)
        self.contact_timer = torch.where(killed_now, 0.0, self.contact_timer)

        if killed_now.any():
            dead_ids = torch.nonzero(killed_now, as_tuple=False).squeeze(1)
            is_targeting_dead = torch.isin(self.locked_target, dead_ids)
            self.locked_target = torch.where(is_targeting_dead, -1, self.locked_target)
            attacker_ids = attacker[killed_now]

            gain = self.MEMORY_GAIN_FACTOR
            self.memory_bonus[attacker_ids] += gain * (self.MEMORY_MAX_BONUS - self.memory_bonus[attacker_ids])
            self.combat_experience[attacker_ids] = torch.clamp(self.combat_experience[attacker_ids] + 0.10, max=1.0)
            self.energy[attacker_ids] = torch.clamp(self.energy[attacker_ids] + 0.15, max=1.0)

    def perform_killing_phase2_cancer_counterattack(self, candidates, dist, t):
        dev = self.device
        cand_is_immune = self.is_immune[candidates.clamp(min=0)]
        local_immune_pressure = (cand_is_immune & (dist < 8.0)).sum(dim=1).float()

        cand_is_cancer = self.is_cancer[candidates.clamp(min=0)]
        dist_cancer = dist.clone().masked_fill(~cand_is_cancer, float('inf'))
        nearest_c_val, nearest_c_idx = torch.min(dist_cancer, dim=1)

        nearest_cancer_particle = torch.gather(candidates, 1, nearest_c_idx.unsqueeze(1)).squeeze(1)

        is_immune_active = self.is_immune & self.active_mask
        in_immune_contact = is_immune_active & (nearest_c_val < self.KILL_RADIUS)
        self.immune_contact_timer = torch.where(in_immune_contact, self.immune_contact_timer + 1, 0.0)

        immune_ready = in_immune_contact & (self.immune_contact_timer >= 5)
        defender_cancer = nearest_cancer_particle.clamp(min=0)
        
        defender_immune_count = local_immune_pressure[defender_cancer]
        immune_pressure_factor = 1.0 / (1.0 + 0.5 * defender_immune_count)

        defender_phenotype = self.phenotype[defender_cancer]
        base_counter_prob = torch.where(defender_phenotype == 0, 0.03, 0.09)
        
        p_counter = base_counter_prob * (0.3 + 0.7 * self.energy[defender_cancer]) * immune_pressure_factor

        counter_roll = torch.rand(self.num_cells, generator=self.gen, device=dev)
        immune_killed = immune_ready & (counter_roll < p_counter)

        self.active_mask = self.active_mask & (~immune_killed)
        self.immune_contact_timer = torch.where(immune_killed, 0.0, self.immune_contact_timer)

    def _write_frame(self, t):
        self.recorded_active_mask[:, t] = self.active_mask

        curr_pos_x = self.positions[:, 0]
        curr_pos_y = self.positions[:, 1]

        self.rec_pos_x[:, t] = curr_pos_x
        self.rec_pos_y[:, t] = curr_pos_y

        if t == 0:
            self.rec_dx_prev[:, 0] = float('nan')
            self.rec_dy_prev[:, 0] = float('nan')
            self.rec_disp_prev[:, 0] = float('nan')
            self.rec_vel_x[:, 0] = float('nan')
            self.rec_vel_y[:, 0] = float('nan')
            self.rec_speed[:, 0] = float('nan')
            
            self.rec_dx_orig[:, 0] = 0.0
            self.rec_dy_orig[:, 0] = 0.0
            self.rec_disp_orig[:, 0] = 0.0
            self.rec_dist_traveled[:, 0] = 0.0
            self.rec_path_efficiency[:, 0] = 1.0
            self.rec_avg_speed[:, 0] = 0.0
        else:
            prev_x = self.rec_pos_x[:, t - 1]
            prev_y = self.rec_pos_y[:, t - 1]

            dx_prev = curr_pos_x - prev_x
            dy_prev = curr_pos_y - prev_y
            disp_prev = torch.sqrt(dx_prev**2 + dy_prev**2)

            vel_x_inst = dx_prev / (self.dt * 10.0)
            vel_y_inst = dy_prev / (self.dt * 10.0)
            speed_inst = torch.sqrt(vel_x_inst**2 + vel_y_inst**2)

            self.cum_dist_traveled += disp_prev

            orig_x = self.initial_pos[:, 0]
            orig_y = self.initial_pos[:, 1]
            dx_orig = curr_pos_x - orig_x
            dy_orig = curr_pos_y - orig_y
            disp_orig = torch.sqrt(dx_orig**2 + dy_orig**2)

            path_eff = torch.where(
                self.cum_dist_traveled > 1e-6,
                disp_orig / self.cum_dist_traveled,
                torch.ones_like(disp_orig)
            )
            path_eff = torch.clamp(path_eff, 0.0, 1.0)

            elapsed_seconds = float(t * self.dt * 10.0)
            avg_speed_val = self.cum_dist_traveled / elapsed_seconds

            self.rec_dx_prev[:, t] = dx_prev
            self.rec_dy_prev[:, t] = dy_prev
            self.rec_disp_prev[:, t] = disp_prev
            self.rec_dx_orig[:, t] = dx_orig
            self.rec_dy_orig[:, t] = dy_orig
            self.rec_disp_orig[:, t] = disp_orig
            self.rec_dist_traveled[:, t] = self.cum_dist_traveled
            self.rec_path_efficiency[:, t] = path_eff
            self.rec_vel_x[:, t] = vel_x_inst
            self.rec_vel_y[:, t] = vel_y_inst
            self.rec_speed[:, t] = speed_inst
            self.rec_avg_speed[:, t] = avg_speed_val

    def step(self, t):
        self._accel.zero_()
        self.update_signals(t)

        candidates, valid, dist, diff = self.find_neighbors()
        gx = torch.clamp((self.positions[:, 0] / self.cell_size).long(), 0, self.grid_w - 1)
        gy = torch.clamp((self.positions[:, 1] / self.cell_size).long(), 0, self.grid_h - 1)

        self.update_immune_cells(candidates, dist, diff, gx, gy, t)
        self.update_cancer_cells(candidates, dist, diff)
        self.apply_environment_noise()
        self.resolve_collisions(candidates, valid, dist, diff)
        self.apply_boundary_forces()

        self.perform_killing_phase1_immune_attack(candidates, dist, t)

        candidates_post1, _, dist_post1, _ = self.find_neighbors(cached_gx=gx, cached_gy=gy)
        self.perform_killing_phase2_cancer_counterattack(candidates_post1, dist_post1, t)

        self.integrate_motion()
        self._write_frame(t)

    def run_simulation(self):
        with torch.inference_mode():
            for t in range(self.timesteps):
                self.step(t)
                if self.device.type == 'cuda' and t % 500 == 0:
                    torch.cuda.empty_cache()

    def export_kinematics_csv(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        T = self.timesteps

        track_ids = self.track_id.cpu().numpy()
        frames = np.arange(T, dtype=np.int32)
        
        act_mask = self.recorded_active_mask.cpu().numpy()
        pos_x = self.rec_pos_x.cpu().numpy()
        pos_y = self.rec_pos_y.cpu().numpy()
        dx_prev = self.rec_dx_prev.cpu().numpy()
        dy_prev = self.rec_dy_prev.cpu().numpy()
        disp_prev = self.rec_disp_prev.cpu().numpy()
        dx_orig = self.rec_dx_orig.cpu().numpy()
        dy_orig = self.rec_dy_orig.cpu().numpy()
        disp_orig = self.rec_disp_orig.cpu().numpy()
        dist_trav = self.rec_dist_traveled.cpu().numpy()
        path_eff = self.rec_path_efficiency.cpu().numpy()
        vel_x = self.rec_vel_x.cpu().numpy()
        vel_y = self.rec_vel_y.cpu().numpy()
        speed = self.rec_speed.cpu().numpy()
        avg_speed = self.rec_avg_speed.cpu().numpy()

        t_cell_rows = []
        cancer_cell_rows = []

        for cell_idx in range(self.num_cells):
            t_id = track_ids[cell_idx]
            is_immune_cell = (t_id <= 100)

            for t_idx in range(T):
                if not act_mask[cell_idx, t_idx]:
                    break

                row = [
                    t_id,
                    frames[t_idx],
                    pos_x[cell_idx, t_idx],
                    pos_y[cell_idx, t_idx],
                    dx_prev[cell_idx, t_idx],
                    dy_prev[cell_idx, t_idx],
                    disp_prev[cell_idx, t_idx],
                    dx_orig[cell_idx, t_idx],
                    dy_orig[cell_idx, t_idx],
                    disp_orig[cell_idx, t_idx],
                    dist_trav[cell_idx, t_idx],
                    path_eff[cell_idx, t_idx],
                    vel_x[cell_idx, t_idx],
                    vel_y[cell_idx, t_idx],
                    speed[cell_idx, t_idx],
                    avg_speed[cell_idx, t_idx],
                ]

                if is_immune_cell:
                    t_cell_rows.append(row)
                else:
                    cancer_cell_rows.append(row)

        columns = [
            "TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y",
            "DX_FROM_PREVIOUS_POINT", "DY_FROM_PREVIOUS_POINT",
            "DISPLACEMENT_FROM_PREVIOUS_POINT", "DX_FROM_ORIGIN",
            "DY_FROM_ORIGIN", "DISPLACEMENT_FROM_ORIGIN",
            "DISTANCE_TRAVELED", "PATH_EFFICIENCY", "VEL_X", "VEL_Y",
            "SPEED", "AVERAGE_SPEED"
        ]

        df_tcell = pd.DataFrame(t_cell_rows, columns=columns)
        df_cancer = pd.DataFrame(cancer_cell_rows, columns=columns)

        tcell_filename = os.path.join(output_dir, f"{self.mode}_T-cell_kinematics.csv")
        cancer_filename = os.path.join(output_dir, f"{self.mode}_Cancer-cell_kinematics.csv")

        df_tcell.to_csv(tcell_filename, index=False)
        df_cancer.to_csv(cancer_filename, index=False)

        return tcell_filename, cancer_filename


# =============================================================================
# 4. DATA LOADING & STRICT RAW PHYSICAL COMPARISON ENGINE
# =============================================================================
def load_dataset_cached(path, name, is_simulation=False, chunksize=200_000):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    preview = pd.read_csv(path, nrows=2)
    cols_present = [col for col in KINEMATIC_COLUMNS if col in preview.columns]
    read_cols = cols_present + (["FRAME"] if "FRAME" in preview.columns else [])
    
    chunks = []
    for chunk in pd.read_csv(path, usecols=read_cols, chunksize=chunksize):
        if is_simulation and "FRAME" in chunk.columns:
            chunk = chunk[chunk["FRAME"] % 6 == 0]
        chunks.append(chunk[cols_present])

    return pd.concat(chunks, ignore_index=True)


def clean_feature_array(df, feature):
    values = pd.to_numeric(df[feature], errors="coerce").dropna().values
    valid_mask = np.isfinite(values)
    if feature in NON_NEGATIVE_FEATURES:
        valid_mask = valid_mask & (values >= 0)
    return values[valid_mask]


def compute_raw_feature_metrics(exp_arr, sim_arr, feature, regime_name):
    """Computes exact 10 concise raw metrics without any Z-scaling or normalization."""
    if len(exp_arr) == 0 or len(sim_arr) == 0:
        return None

    e_mean, s_mean = float(np.mean(exp_arr)), float(np.mean(sim_arr))
    e_med, s_med = float(np.median(exp_arr)), float(np.median(sim_arr))
    e_std, s_std = float(np.std(exp_arr)), float(np.std(sim_arr))

    w_dist = float(stats.wasserstein_distance(exp_arr, sim_arr))
    ks_stat = float(stats.ks_2samp(exp_arr, sim_arr).statistic)

    return {
        "Regime": regime_name,
        "Feature": feature,
        "Units": FEATURE_UNITS[feature],
        "Exp_Mean": round(e_mean, 4),
        "Sim_Mean": round(s_mean, 4),
        "Abs_Mean_Diff": round(abs(s_mean - e_mean), 4),
        "Exp_Median": round(e_med, 4),
        "Sim_Median": round(s_med, 4),
        "Abs_Median_Diff": round(abs(s_med - e_med), 4),
        "Exp_Std": round(e_std, 4),
        "Sim_Std": round(s_std, 4),
        "Abs_Std_Diff": round(abs(s_std - e_std), 4),
        "Wasserstein": round(w_dist, 4),
        "KS_Stat": round(ks_stat, 4),
    }


def plot_14_feature_raw_comparison(exp_df, sim_df, exp_name, sim_name, param_label, output_path):
    n_features = len(KINEMATIC_COLUMNS)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.8 * nrows), dpi=150)
    axes = np.asarray(axes).flatten()

    for i, feature in enumerate(KINEMATIC_COLUMNS):
        ax = axes[i]
        unit = FEATURE_UNITS[feature]

        exp_raw = clean_feature_array(exp_df, feature)
        sim_raw = clean_feature_array(sim_df, feature)

        if len(exp_raw) == 0 or len(sim_raw) == 0:
            ax.set_title(f"{feature}\n(no valid data)")
            ax.axis("off")
            continue

        combined = np.concatenate([exp_raw, sim_raw])
        min_v, max_v = np.percentile(combined, [0.05, 99.95])

        ax.hist(
            exp_raw,
            bins=60,
            range=(min_v, max_v) if min_v < max_v else None,
            density=True,
            alpha=0.55,
            color="blue",
            label="Experimental"
        )
        ax.hist(
            sim_raw,
            bins=60,
            range=(min_v, max_v) if min_v < max_v else None,
            density=True,
            alpha=0.55,
            color="red",
            label="Simulation"
        )

        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Value ({unit})" if unit != "dimensionless" else "Value (dimensionless)", fontsize=9.5)
        ax.set_ylabel("Probability Density", fontsize=9.5)
        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{exp_name} vs {sim_name}\n"
        f"Kinematic Distribution Comparison (Raw Physical Units) [{param_label}]",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_four_regime_raw_summary(regime_data_map, param_label, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    panels = [
        ("Killing Cancer", axes[0, 0]),
        ("Killing T-Cell", axes[0, 1]),
        ("Non-Killing Cancer", axes[1, 0]),
        ("Non-Killing T-Cell", axes[1, 1]),
    ]

    for regime_name, ax in panels:
        exp_df, sim_df = regime_data_map[regime_name]
        
        key_feat = "DX_FROM_ORIGIN"  # Using displacement from origin as a representative feature for summary
        exp_v = clean_feature_array(exp_df, key_feat)
        sim_v = clean_feature_array(sim_df, key_feat)

        if len(exp_v) > 0 and len(sim_v) > 0:
            combined = np.concatenate([exp_v, sim_v])
            min_v, max_v = np.percentile(combined, [0.05, 99.95])

            ax.hist(exp_v, bins=50, range=(min_v, max_v) if min_v < max_v else None,
                    density=True, alpha=0.55, color="blue", label="Experimental")
            ax.hist(sim_v, bins=50, range=(min_v, max_v) if min_v < max_v else None,
                    density=True, alpha=0.55, color="red", label="Simulation")
            
            ax.set_title(f"{regime_name} — Raw DISPLACEMENT from X[0] (µm)", fontsize=12, fontweight="bold")
            ax.set_xlabel("DISPLACEMENT from X[0](µm)")
            ax.set_ylabel("Probability Density")
            ax.grid(alpha=0.2)
            ax.legend(frameon=True)
        else:
            ax.set_title(f"{regime_name} (no valid data)")

    fig.suptitle(
        f"4-Regime Representative Overview (Raw Physical Units) [{param_label}]\n"
        "(Refer to individual 14-feature figures for complete breakdown)",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# =============================================================================
# 5. PARAMETER TUNING MAIN PIPELINE
# =============================================================================
def format_param_string(param_name, param_val):
    if isinstance(param_val, tuple):
        val_str = f"{param_val[0]}_{param_val[1]}"
        label_str = f"{param_name} = {param_val[0]} – {param_val[1]}"
    else:
        val_str = f"{param_val}"
        label_str = f"{param_name} = {param_val}"
    folder_name = f"{param_name}_{val_str}"
    return folder_name, label_str


def main():
    parser = argparse.ArgumentParser(description="Cell Kinematics Parameter Tuning Pipeline (Raw Physical Units)")
    parser.add_argument("--parameter", type=str, default=TUNING_PARAMETER, help="Simulator parameter to tune")
    args = parser.parse_args()

    param_to_tune = args.parameter
    values_to_sweep = TUNING_VALUES
    total_iters = len(values_to_sweep)

    param_base_dir = os.path.join(BASE_OUTPUT_DIR, param_to_tune)
    os.makedirs(param_base_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("      CELL KINEMATICS PARAMETER TUNING & VISUAL SWEEP PIPELINE       ")
    print("                      (STRICT RAW PHYSICAL UNITS)                    ")
    print("=" * 70)
    print(f"Parameter to Tune : {param_to_tune}")
    print(f"Total Iterations  : {total_iters}")
    print(f"Hardware Device   : {DEVICE}")
    print(f"Output Directory  : {os.path.abspath(param_base_dir)}")
    print("=" * 70 + "\n")

    # Step 1: Pre-load all 4 Experimental Datasets ONCE
    print("--- PRE-LOADING EXPERIMENTAL DATASETS (CACHED) ---")
    exp_datasets = {
        "Killing Cancer": load_dataset_cached(EXPERIMENTAL_KILLING_CANCER, "Experimental Killing Cancer"),
        "Killing T-Cell": load_dataset_cached(EXPERIMENTAL_KILLING_TCELL, "Experimental Killing T-Cell"),
        "Non-Killing Cancer": load_dataset_cached(EXPERIMENTAL_NONKILLING_CANCER, "Experimental Non-Killing Cancer"),
        "Non-Killing T-Cell": load_dataset_cached(EXPERIMENTAL_NONKILLING_TCELL, "Experimental Non-Killing T-Cell"),
    }
    print("Experimental datasets loaded and cached in memory.\n")

    # Step 2: Parameter Sweep Execution Loop
    for idx, param_val in enumerate(values_to_sweep, start=1):
        folder_name, label_str = format_param_string(param_to_tune, param_val)
        iter_output_dir = os.path.join(param_base_dir, folder_name)
        os.makedirs(iter_output_dir, exist_ok=True)

        print("=" * 70)
        print(f"ITERATION {idx} / {total_iters}")
        print(f"Setting: {label_str}")
        print(f"Destination: {iter_output_dir}")
        print("=" * 70)

        # 2A: Configure and Run Simulator
        print("Running Simulator for Killing Regime...")
        sim_kill = CellSimulation(
            num_immune=100, num_cancer=100,
            mode='killing', seed=RANDOM_SEED,
            param_override={param_to_tune: param_val}
        )
        sim_kill.run_simulation()
        tcell_kill_csv, cancer_kill_csv = sim_kill.export_kinematics_csv(iter_output_dir)

        print("Running Simulator for Non-Killing Regime...")
        sim_nonkill = CellSimulation(
            num_immune=100, num_cancer=100,
            mode='non-killing', seed=RANDOM_SEED,
            param_override={param_to_tune: param_val}
        )
        sim_nonkill.run_simulation()
        tcell_nonkill_csv, cancer_nonkill_csv = sim_nonkill.export_kinematics_csv(iter_output_dir)

        # 2B: Load Simulation Outputs (Filtered to FRAME % 6 == 0)
        print("\nLoading Simulation Outputs...")
        sim_datasets = {
            "Killing Cancer": load_dataset_cached(cancer_kill_csv, "Sim Killing Cancer", is_simulation=True),
            "Killing T-Cell": load_dataset_cached(tcell_kill_csv, "Sim Killing T-Cell", is_simulation=True),
            "Non-Killing Cancer": load_dataset_cached(cancer_nonkill_csv, "Sim Non-Killing Cancer", is_simulation=True),
            "Non-Killing T-Cell": load_dataset_cached(tcell_nonkill_csv, "Sim Non-Killing T-Cell", is_simulation=True),
        }

        # 2C: Generate Four 14-Feature Raw Figures & Compute Concise 10-Metric Table
        print("Generating 14-Feature Histograms & Concise Metrics Table...")
        regimes = [
            ("Killing Cancer", "killing_cancer.png"),
            ("Killing T-Cell", "killing_tcell.png"),
            ("Non-Killing Cancer", "nonkilling_cancer.png"),
            ("Non-Killing T-Cell", "nonkilling_tcell.png"),
        ]

        regime_data_map = {}
        metrics_rows = []

        for r_name, fig_filename in regimes:
            e_df = exp_datasets[r_name]
            s_df = sim_datasets[r_name]
            regime_data_map[r_name] = (e_df, s_df)

            fig_path = os.path.join(iter_output_dir, fig_filename)
            plot_14_feature_raw_comparison(
                e_df, s_df,
                f"Experimental {r_name}", f"Simulated {r_name}",
                label_str, fig_path
            )

            # Calculate raw metrics for all 14 features
            for feat in KINEMATIC_COLUMNS:
                e_arr = clean_feature_array(e_df, feat)
                s_arr = clean_feature_array(s_df, feat)
                res = compute_raw_feature_metrics(e_arr, s_arr, feat, r_name)
                if res:
                    metrics_rows.append(res)

        # Export concise metrics CSV
        metrics_df = pd.DataFrame(metrics_rows)
        csv_metrics_path = os.path.join(iter_output_dir, "raw_metrics_summary.csv")
        metrics_df.to_csv(csv_metrics_path, index=False)
        print(f"Concise metrics table saved: {csv_metrics_path}")

        # 2D: Generate 4-Regime Raw Summary Overview
        summary_fig_path = os.path.join(iter_output_dir, "00_all_regimes_summary_overview.png")
        plot_four_regime_raw_summary(regime_data_map, label_str, summary_fig_path)

        print(f"Completed Iteration {idx} / {total_iters}")
        print(f"All outputs saved to:\n  {iter_output_dir}\n")

    print("\n" + "=" * 70)
    print("                    PARAMETER SWEEP COMPLETE                         ")
    print("=" * 70)
    print(f"Parameter Tuned : {param_to_tune}")
    print(f"Total Runs      : {total_iters}")
    print(f"Master Directory: {os.path.abspath(param_base_dir)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()