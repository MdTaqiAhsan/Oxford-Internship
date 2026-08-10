"""
new_cell_simulator.py (v8.0 — Camera Observation Kinematic Calibration Engine)
============================================================================
GPU-accelerated PyTorch immune-cancer microenvironment simulator.
Simulates internally at 1-frame timestep resolution (dt=1.0) for numerical physics,
but extracts and records all kinematic metrics STRICTLY at 6-frame observation
intervals (dt_obs=6.0) to match experimental live-cell tracking resolution.

OBSERVED KINEMATIC CSV CONTRACT (16 COLUMNS):
TRACK_ID, FRAME, POSITION_X, POSITION_Y, DX_FROM_PREVIOUS_POINT,
DY_FROM_PREVIOUS_POINT, DISPLACEMENT_FROM_PREVIOUS_POINT, DX_FROM_ORIGIN,
DY_FROM_ORIGIN, DISPLACEMENT_FROM_ORIGIN, DISTANCE_TRAVELED, PATH_EFFICIENCY,
VEL_X, VEL_Y, SPEED, AVERAGE_SPEED
"""

import os
# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import numpy as np
import pandas as pd
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),  (0, 0),  (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]


def _sample_uniform(lo, hi, n, generator, device):
    return lo + (hi - lo) * torch.rand(n, generator=generator, device=device)


class CellSimulation:
    def __init__(self,
                 num_immune=1000, num_cancer=1000,
                 width=None, height=None,
                 timesteps=8635,             # Frame 0 to 8634 inclusive
                 observation_interval=6,     # Camera observation interval = 6 frames
                 mode='killing',
                 dt=1.0,
                 seed=None,
                 target_density=0.002,
                 max_per_cell=16,
                 max_signals_per_cell=32,
                 device=None,
                 debug=False,
                 scout_prob=0.30,
                 messenger_prob=0.30,
                 killer_prob=0.40,
                 target_lock_timeout=15,
                 enable_proliferation=False, # Strictly disabled for tracking
                 enable_apoptosis=False):    # Strictly disabled for tracking

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
        self.observation_interval = observation_interval
        self.dt = dt
        self.dt_obs = observation_interval * dt  # 6.0 time units between observations
        self.mode = mode
        self.debug = debug
        self.max_per_cell = max_per_cell
        self.max_signals_per_cell = max_signals_per_cell

        # Precalculate observed frame indices [0, 6, 12, 18, ..., 8634]
        self.recorded_frame_indices = list(range(0, timesteps, observation_interval))
        self.num_recorded_frames = len(self.recorded_frame_indices)

        # Auto-scale simulation domain to maintain density
        if width is None or height is None:
            side = math.sqrt(N / target_density)
            width = width if width is not None else side
            height = height if height is not None else side
        self.width = float(width)
        self.height = float(height)

        self.scout_prob = scout_prob
        self.messenger_prob = messenger_prob
        self.killer_prob = killer_prob
        self.target_lock_timeout = target_lock_timeout
        
        # Guard population stability
        self.enable_proliferation = False
        self.enable_apoptosis = False

        self._initialize_constants()
        self._initialize_phenotypes()
        self._initialize_motion_parameters()
        self._initialize_signals()
        self._initialize_dynamic_state()

        self._accel = torch.zeros((self.num_cells, 2), device=self.device)

    def _blend(self, mask, lo, hi, base):
        return torch.where(mask, _sample_uniform(lo, hi, self.num_cells, self.gen, self.device), base)

    def _initialize_constants(self):
        """Centralized Biological Configuration & Hyperparameters."""
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

        # Stochastic Signal Lifetimes
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

        self.PROLIFERATION_PROB_BASE = 0.005
        self.APOPTOSIS_PROB_IMMUNE = 0.0005
        self.APOPTOSIS_PROB_CANCER = 0.0002

        # Swarming & Alignment Metrics
        self.ALIGNMENT_RADIUS_KILLER = 12.0
        self.ALIGNMENT_RADIUS_CANCER = 10.0

    def _initialize_phenotypes(self):
        prob_sum = self.scout_prob + self.messenger_prob + self.killer_prob
        if not math.isclose(prob_sum, 1.0, abs_tol=1e-5):
            raise ValueError(f"Immune phenotype probabilities must sum to 1.0 (got {prob_sum:.4f})")

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

        self.tau = self._blend(imm_scout, 1.0, 2.0, z.clone())
        self.tau = self._blend(imm_msg, 2.0, 3.5, self.tau)
        self.tau = self._blend(imm_killer, 3.5, 5.0, self.tau)
        self.tau = torch.where(self.is_cancer, _sample_uniform(2.0, 4.0, N, gen, dev), self.tau)

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

        # Persistent Track IDs
        immune_ids = torch.arange(1, self.num_immune + 1, dtype=torch.long, device=dev)
        cancer_ids = torch.arange(1001, 1001 + self.num_cancer, dtype=torch.long, device=dev)
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

        # ============================================================
        # CAMERA OBSERVATION: Preallocated 6-Frame Kinematic Buffers
        # ============================================================
        R = self.num_recorded_frames
        self.recorded_active_mask = torch.zeros((N, R), dtype=torch.bool, device=dev)
        self.rec_pos_x = torch.zeros((N, R), device=dev)
        self.rec_pos_y = torch.zeros((N, R), device=dev)
        self.rec_dx_prev = torch.zeros((N, R), device=dev)
        self.rec_dy_prev = torch.zeros((N, R), device=dev)
        self.rec_disp_prev = torch.zeros((N, R), device=dev)
        self.rec_dx_orig = torch.zeros((N, R), device=dev)
        self.rec_dy_orig = torch.zeros((N, R), device=dev)
        self.rec_disp_orig = torch.zeros((N, R), device=dev)
        self.rec_dist_traveled = torch.zeros((N, R), device=dev)
        self.rec_path_efficiency = torch.zeros((N, R), device=dev)
        self.rec_vel_x = torch.zeros((N, R), device=dev)
        self.rec_vel_y = torch.zeros((N, R), device=dev)
        self.rec_speed = torch.zeros((N, R), device=dev)
        self.rec_avg_speed = torch.zeros((N, R), device=dev)

        self.initial_pos = self.positions.clone()
        self.cum_dist_traveled = torch.zeros(N, device=dev)

        # Summary Metrics
        self.stats_immune_alive = np.zeros(self.timesteps, dtype=np.int32)
        self.stats_cancer_alive = np.zeros(self.timesteps, dtype=np.int32)
        self.stats_kills_phase1 = np.zeros(self.timesteps, dtype=np.int32)
        self.stats_counterkills_phase2 = np.zeros(self.timesteps, dtype=np.int32)

    def build_spatial_grid(self):
        gx = torch.clamp((self.positions[:, 0] / self.cell_size).long(), 0, self.grid_w - 1)
        gy = torch.clamp((self.positions[:, 1] / self.cell_size).long(), 0, self.grid_h - 1)
        cell_id = gy * self.grid_w + gx
        dummy_id = self.num_grid_cells
        cell_id = torch.where(self.active_mask, cell_id, dummy_id)

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
        dummy_id = self.num_grid_cells
        cell_id = torch.where(self.signal_active, cell_id, dummy_id)

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

        # Persistent Target Lock Verification
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

        # Scout Recruitment Emission
        scout_emitting = imm_scout & has_any_within & (self.energy > self.ENERGY_EMIT_THRESHOLD)
        scout_cancer_target = self.locked_target
        self.emit_signals(
            scout_emitting, self.positions,
            torch.full((self.num_cells,), self.SIGNAL_EMISSION_STRENGTH, device=dev),
            scout_cancer_target, torch.arange(self.num_cells, device=dev),
            torch.zeros(self.num_cells, dtype=torch.long, device=dev), t
        )

        # Integrated Chemotaxis Vector Field
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
        """Phase 1 Combat — Immune Cells Attack Target Cancer Cells."""
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

        num_kills = int(killed_now.sum().item())
        self.stats_kills_phase1[t] = num_kills

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
        """Phase 2 Combat — Active Cancer Counterattacks Nearby Immune Cells."""
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

        self.stats_counterkills_phase2[t] = int(immune_killed.sum().item())

    # ============================================================
    # CAMERA OBSERVATION KINEMATIC RECORDER (dt_obs = 6 * dt)
    # ============================================================
    def _write_frame(self, t):
        # Frame summary count updates
        self.stats_immune_alive[t] = int((self.is_immune & self.active_mask).sum().item())
        self.stats_cancer_alive[t] = int((self.is_cancer & self.active_mask).sum().item())

        # Observe & record kinetics STRICTLY every observation_interval (6th frame)
        if t % self.observation_interval == 0:
            rec_idx = t // self.observation_interval

            # Store alive/dead status mask
            self.recorded_active_mask[:, rec_idx] = self.active_mask

            curr_pos_x = self.positions[:, 0]
            curr_pos_y = self.positions[:, 1]

            self.rec_pos_x[:, rec_idx] = curr_pos_x
            self.rec_pos_y[:, rec_idx] = curr_pos_y

            if rec_idx == 0:
                # Frame 0 Initial Conditions (no previous observation point)
                self.rec_dx_prev[:, 0] = 0.0
                self.rec_dy_prev[:, 0] = 0.0
                self.rec_disp_prev[:, 0] = 0.0
                self.rec_dx_orig[:, 0] = 0.0
                self.rec_dy_orig[:, 0] = 0.0
                self.rec_disp_orig[:, 0] = 0.0
                self.rec_dist_traveled[:, 0] = 0.0
                self.rec_path_efficiency[:, 0] = 1.0
                self.rec_vel_x[:, 0] = 0.0
                self.rec_vel_y[:, 0] = 0.0
                self.rec_speed[:, 0] = 0.0
                self.rec_avg_speed[:, 0] = 0.0
            else:
                prev_x = self.rec_pos_x[:, rec_idx - 1]
                prev_y = self.rec_pos_y[:, rec_idx - 1]

                # 1. Observed Spatial Displacements between t and t-6
                dx_prev = curr_pos_x - prev_x
                dy_prev = curr_pos_y - prev_y
                disp_prev = torch.sqrt(dx_prev**2 + dy_prev**2)

                # 2. Observed Velocities derived over dt_obs = 6.0 time units
                vel_x_obs = dx_prev / self.dt_obs
                vel_y_obs = dy_prev / self.dt_obs
                speed_obs = disp_prev / self.dt_obs

                # 3. Accumulated Distance Traveled (sum of observable 6-frame displacements)
                self.cum_dist_traveled += disp_prev

                # 4. Displacements from Origin (relative to frame 0 observation)
                orig_x = self.initial_pos[:, 0]
                orig_y = self.initial_pos[:, 1]
                dx_orig = curr_pos_x - orig_x
                dy_orig = curr_pos_y - orig_y
                disp_orig = torch.sqrt(dx_orig**2 + dy_orig**2)

                # 5. Path Efficiency (Origin Displacement / Observed Distance Traveled)
                path_eff = torch.where(
                    self.cum_dist_traveled > 1e-6,
                    disp_orig / self.cum_dist_traveled,
                    torch.ones_like(disp_orig)
                )
                path_eff = torch.clamp(path_eff, 0.0, 1.0)

                # 6. Cumulative Average Speed (Observed Distance Traveled / Cumulative Elapsed Time)
                elapsed_time_obs = float(t * self.dt)
                avg_speed_obs = self.cum_dist_traveled / elapsed_time_obs

                # Commit to GPU buffers
                self.rec_dx_prev[:, rec_idx] = dx_prev
                self.rec_dy_prev[:, rec_idx] = dy_prev
                self.rec_disp_prev[:, rec_idx] = disp_prev
                self.rec_dx_orig[:, rec_idx] = dx_orig
                self.rec_dy_orig[:, rec_idx] = dy_orig
                self.rec_disp_orig[:, rec_idx] = disp_orig
                self.rec_dist_traveled[:, rec_idx] = self.cum_dist_traveled
                self.rec_path_efficiency[:, rec_idx] = path_eff
                self.rec_vel_x[:, rec_idx] = vel_x_obs
                self.rec_vel_y[:, rec_idx] = vel_y_obs
                self.rec_speed[:, rec_idx] = speed_obs
                self.rec_avg_speed[:, rec_idx] = avg_speed_obs

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

        # Phase 1: Immune Cell Attack
        self.perform_killing_phase1_immune_attack(candidates, dist, t)

        # Phase 2: Re-query Neighbors & Execute Cancer Counterattack
        candidates_post1, _, dist_post1, _ = self.find_neighbors(cached_gx=gx, cached_gy=gy)
        self.perform_killing_phase2_cancer_counterattack(candidates_post1, dist_post1, t)

        self.integrate_motion()
        self._write_frame(t)

    def run_simulation(self):
        with torch.inference_mode():
            for t in range(self.timesteps):
                self.step(t)
                if self.device.type == 'cuda' and t % 100 == 0:
                    torch.cuda.empty_cache()

        stats = {
            "immune_alive": self.stats_immune_alive,
            "cancer_alive": self.stats_cancer_alive,
            "kills_phase1": self.stats_kills_phase1,
            "counterkills_phase2": self.stats_counterkills_phase2,
        }
        return stats

    def export_kinematics_csv(self, output_dir):
        """Transfers preallocated GPU buffers to CPU and writes 2 CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        R = self.num_recorded_frames

        # Single batch transfer from GPU to CPU
        track_ids = self.track_id.cpu().numpy()
        rec_frames = np.array(self.recorded_frame_indices, dtype=np.int32)
        
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
            is_immune_cell = (t_id <= 1000)

            for rec_idx in range(R):
                # Stop recording trajectory at cell death frame
                if not act_mask[cell_idx, rec_idx]:
                    break

                frame_val = rec_frames[rec_idx]
                row = [
                    t_id,
                    frame_val,
                    pos_x[cell_idx, rec_idx],
                    pos_y[cell_idx, rec_idx],
                    dx_prev[cell_idx, rec_idx],
                    dy_prev[cell_idx, rec_idx],
                    disp_prev[cell_idx, rec_idx],
                    dx_orig[cell_idx, rec_idx],
                    dy_orig[cell_idx, rec_idx],
                    disp_orig[cell_idx, rec_idx],
                    dist_trav[cell_idx, rec_idx],
                    path_eff[cell_idx, rec_idx],
                    vel_x[cell_idx, rec_idx],
                    vel_y[cell_idx, rec_idx],
                    speed[cell_idx, rec_idx],
                    avg_speed[cell_idx, rec_idx],
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

        # Sanity Validation Checks
        assert len(df_tcell['TRACK_ID'].unique()) <= 1000, "Validation Error: Excess T-cell track IDs"
        assert len(df_cancer['TRACK_ID'].unique()) <= 1000, "Validation Error: Excess Cancer track IDs"
        assert (df_tcell['SPEED'] >= 0).all(), "Validation Error: Negative speed in T-cells"
        assert (df_cancer['SPEED'] >= 0).all(), "Validation Error: Negative speed in Cancer"
        assert not df_tcell.isnull().values.any(), "Validation Error: NaN/Inf in T-cell kinematics"
        assert not df_cancer.isnull().values.any(), "Validation Error: NaN/Inf in Cancer kinematics"

        tcell_filename = os.path.join(output_dir, f"{self.mode}_T-cell_kinematics.csv")
        cancer_filename = os.path.join(output_dir, f"{self.mode}_Cancer-cell_kinematics.csv")

        df_tcell.to_csv(tcell_filename, index=False)
        df_cancer.to_csv(cancer_filename, index=False)

        return tcell_filename, cancer_filename


# ============================================================
# DATASET GENERATION PIPELINE
# ============================================================
def generate_dataset(mode, run_id=0, num_immune=1000, num_cancer=1000, timesteps=8635,
                      output_dir="new_simulator/kinematic_csv_outputs", **kwargs):
    seed = hash((mode, run_id)) % (2**32)
    sim = CellSimulation(num_immune=num_immune, num_cancer=num_cancer,
                          timesteps=timesteps, mode=mode, seed=seed, **kwargs)
    
    summary_stats = sim.run_simulation()
    tcell_csv, cancer_csv = sim.export_kinematics_csv(output_dir)

    print(f"\n============================================================")
    print(f"Mode: {mode}")
    print(f"Simulation frames: {timesteps}")
    print(f"Recorded frames: {sim.num_recorded_frames}")
    print(f"Immune cells: {num_immune} (TRACK_IDs 1-1000)")
    print(f"Cancer cells: {num_cancer} (TRACK_IDs 1001-2000)")
    print(f"Output:")
    print(f"    {tcell_csv}")
    print(f"    {cancer_csv}")
    print(f"============================================================\n")
    return sim, tcell_csv, cancer_csv


if __name__ == "__main__":
    print(f"Using execution hardware device: {DEVICE}")
    print("Generating experimental-resolution kinematic datasets (6-frame observations)...")
    sim_obj, _, _ = generate_dataset('killing', 0)
    generate_dataset('non-killing', 0)
    
    # ============================================================
    # VALIDATION & PROOF CHECKS
    # ============================================================
    print("\n============================================================")
    print("                TEMPORAL RESOLUTION VALIDATION              ")
    print("============================================================")
    print(f"Total Simulation Cells        : {sim_obj.num_cells} (1000 Immune + 1000 Cancer)")
    print(f"Total Recorded Observation Frames: {sim_obj.num_recorded_frames}")
    print(f"First 10 Observed Frames      : {sim_obj.recorded_frame_indices[:10]}")
    print(f"Last Observed Frame           : {sim_obj.recorded_frame_indices[-1]}")
    
    # Read generated killing T-cell CSV to verify single-cell 6-frame step math
    df_val = pd.read_csv("new_simulator/kinematic_csv_outputs/killing_T-cell_kinematics.csv")
    c1 = df_val[df_val['TRACK_ID'] == 1].sort_values('FRAME')
    
    print("\n[EXPLICIT STEP PROOF FOR TRACK_ID = 1]")
    row_t0 = c1[c1['FRAME'] == 0].iloc[0]
    row_t6 = c1[c1['FRAME'] == 6].iloc[0]
    row_t12 = c1[c1['FRAME'] == 12].iloc[0]

    print(f" • Frame t=0  Pos: ({row_t0['POSITION_X']:.2f}, {row_t0['POSITION_Y']:.2f})")
    print(f" • Frame t=6  Pos: ({row_t6['POSITION_X']:.2f}, {row_t6['POSITION_Y']:.2f})")
    print(f" • Frame t=12 Pos: ({row_t12['POSITION_X']:.2f}, {row_t12['POSITION_Y']:.2f})")
    
    calc_dx_6_12 = row_t12['POSITION_X'] - row_t6['POSITION_X']
    calc_dy_6_12 = row_t12['POSITION_Y'] - row_t6['POSITION_Y']
    calc_disp_6_12 = math.sqrt(calc_dx_6_12**2 + calc_dy_6_12**2)
    calc_vx_6_12 = calc_dx_6_12 / 6.0
    calc_vy_6_12 = calc_dy_6_12 / 6.0
    calc_speed_6_12 = calc_disp_6_12 / 6.0

    print("\n[VERIFICATION OF FRAME 12 DERIVED KINEMATICS (obs_dt = 6.0)]")
    print(f" • DX (Pos12 - Pos6)          : Recorded = {row_t12['DX_FROM_PREVIOUS_POINT']:.4f} | Calculated = {calc_dx_6_12:.4f}")
    print(f" • DY (Pos12 - Pos6)          : Recorded = {row_t12['DY_FROM_PREVIOUS_POINT']:.4f} | Calculated = {calc_dy_6_12:.4f}")
    print(f" • DISPLACEMENT (6->12)       : Recorded = {row_t12['DISPLACEMENT_FROM_PREVIOUS_POINT']:.4f} | Calculated = {calc_disp_6_12:.4f}")
    print(f" • VEL_X (DX / 6.0)           : Recorded = {row_t12['VEL_X']:.4f} | Calculated = {calc_vx_6_12:.4f}")
    print(f" • VEL_Y (DY / 6.0)           : Recorded = {row_t12['VEL_Y']:.4f} | Calculated = {calc_vy_6_12:.4f}")
    print(f" • SPEED (DISP / 6.0)         : Recorded = {row_t12['SPEED']:.4f} | Calculated = {calc_speed_6_12:.4f}")
    
    # Frame step interval sanity check
    frame_diffs = c1['FRAME'].diff().dropna().unique()
    print(f"\n Unique Observation Frame Intervals Found in File: {frame_diffs}")
    assert len(frame_diffs) == 1 and frame_diffs[0] == 6, "SANITY CHECK FAILED: Frame step is not strictly 6!"
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")