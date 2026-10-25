"""
cell_simulator.py
============================================================================
Continuous-Frame Cell Kinematics Simulator (PyTorch GPU/CPU)
JSON-Calibrated Configuration Engine (Supports Dual Regime Ingestion)
----------------------------------------------------------------------------
Features:
  - Dynamically loads and binds optimal parameters from JSON artifacts:
      * optimal_parameters_killing_0to5hrs.json
      * optimal_parameters_non-killing_0to5hrs.json
  - Configures biophysical constants, sensing ranges, tau distributions, 
    noise scales, phenotype ratios, and combat dynamics.
  - Complete 17-feature in-memory kinematic recording & CSV export.
============================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import json
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
                 width=1536.0, height=1536.0,
                 timesteps=8635,
                 mode='killing',
                 dt=1.0,
                 seed=None,
                 params_dict=None,
                 target_density=0.002,
                 max_per_cell=16,
                 max_signals_per_cell=32,
                 device=None,
                 debug=False,
                 target_lock_timeout=15,
                 enable_proliferation=False,
                 enable_apoptosis=False):

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
        self.debug = debug
        self.max_per_cell = max_per_cell
        self.max_signals_per_cell = max_signals_per_cell

        self.width = float(width) if width is not None else 1536.0
        self.height = float(height) if height is not None else 1536.0

        self.target_lock_timeout = target_lock_timeout
        self.enable_proliferation = False
        self.enable_apoptosis = False

        # Apply parameters from injected JSON/dict, otherwise use calibrated defaults
        self.params = params_dict or {}

        # Phenotype ratios
        self.scout_prob = self.params.get("scout_prob", 0.254238)
        self.messenger_prob = self.params.get("messenger_prob", 0.121813)
        self.killer_prob = self.params.get("killer_prob", max(0.05, 1.0 - (self.scout_prob + self.messenger_prob)))

        self._initialize_constants()
        self._initialize_phenotypes()
        self._initialize_motion_parameters()
        self._initialize_signals()
        self._initialize_dynamic_state()

        self._accel = torch.zeros((self.num_cells, 2), device=self.device)

    def _blend(self, mask, lo, hi, base):
        return torch.where(mask, _sample_uniform(lo, hi, self.num_cells, self.gen, self.device), base)

    def _initialize_constants(self):
        p = self.params

        # Base speeds and combat recognition
        if self.mode == 'killing':
            self.IMMUNE_BASE_MEAN = p.get("IMMUNE_BASE_MEAN", 4.591852)
            self.IMMUNE_BASE_STD = p.get("IMMUNE_BASE_STD", 1.542007)
            self.RECOG_BASE = (0.85, 0.60, 0.95)
            self.SIGNAL_EMISSION_STRENGTH = p.get("SIGNAL_EMISSION_STRENGTH", 0.814516)
        else:
            self.IMMUNE_BASE_MEAN = p.get("IMMUNE_BASE_MEAN", 5.200000)
            self.IMMUNE_BASE_STD = p.get("IMMUNE_BASE_STD", 1.600000)
            self.RECOG_BASE = (0.25, 0.15, 0.30)
            self.SIGNAL_EMISSION_STRENGTH = p.get("SIGNAL_EMISSION_STRENGTH", 0.350000)

        # Phenotype speed multipliers and cancer speed bounds
        self.SPEED_MULTS = (
            p.get("speed_mult_scout", 1.148022),
            p.get("speed_mult_msg", 1.218243),
            p.get("speed_mult_killer", 0.898411)
        )
        self.CAN_SESSILE_SPEED = (
            p.get("can_sessile_min", 0.144107),
            p.get("can_sessile_max", 1.992687)
        )
        self.CAN_EVASIVE_SPEED = (
            p.get("can_evasive_min", 1.632570),
            p.get("can_evasive_max", 3.063596)
        )

        # Sensing ranges (µm)
        self.SENSING_SCOUT = (p.get("sensing_scout_min", 16.845729), p.get("sensing_scout_max", 32.089831))
        self.SENSING_MESSENGER = (p.get("sensing_msg_min", 15.806944), p.get("sensing_msg_max", 19.178486))
        self.SENSING_KILLER = (p.get("sensing_killer_min", 7.606684), p.get("sensing_killer_max", 16.293754))
        self.SENSING_CANCER = (p.get("sensing_can_min", 9.279925), p.get("sensing_can_max", 14.819810))
        self.EVASIVE_HIDE_PENALTY = 0.40

        # Signaling & chemotaxis
        self.MAX_SIGNALS = 2000
        self.SIGNAL_DECAY_RATE = p.get("SIGNAL_DECAY_RATE", 0.113598)
        self.SIGNAL_SENSING_RADIUS = p.get("SIGNAL_SENSING_RADIUS", 35.437976)
        self.SIGNAL_MAX_CAP = 3.0
        self.AMPLIFY_COOLDOWN_STEPS = 5
        self.CHEMOTAXIS_EPSILON = 2.0

        self.SIGNAL_LIFETIME_SCOUT_MEAN = 25.0
        self.SIGNAL_LIFETIME_SCOUT_STD = 5.0
        self.SIGNAL_LIFETIME_MSG_MEAN = 15.0
        self.SIGNAL_LIFETIME_MSG_STD = 3.0

        # Combat & spatial interaction
        self.KILL_RADIUS = p.get("KILL_RADIUS", 3.163265)
        self.ENGAGE_STEPS_REQUIRED = 4
        self.MIN_SEPARATION = p.get("MIN_SEPARATION", 1.330658)
        self.BOUNDARY_MARGIN = 8.0
        self.KILL_RATES = (0.01, 0.00, 0.35 if self.mode == 'killing' else 0.0175)

        # Memory dynamics
        self.MEMORY_DECAY_RATE = 0.005
        self.MEMORY_GAIN_FACTOR = p.get("MEMORY_GAIN_FACTOR", 0.183850)
        self.MEMORY_MAX_BONUS = p.get("MEMORY_MAX_BONUS", 0.319025)

        # Energy dynamics
        self.ENERGY_DRAIN_MOVE = p.get("ENERGY_DRAIN_MOVE", 0.000546)
        self.ENERGY_DRAIN_CHASE = p.get("ENERGY_DRAIN_CHASE", 0.003208)
        self.ENERGY_DRAIN_COMBAT = p.get("ENERGY_DRAIN_COMBAT", 0.029167)
        self.ENERGY_RECOVER_REST = p.get("ENERGY_RECOVER_REST", 0.023428)
        self.ENERGY_EMIT_THRESHOLD = 0.15

        self.PROLIFERATION_PROB_BASE = 0.005
        self.APOPTOSIS_PROB_IMMUNE = 0.0005
        self.APOPTOSIS_PROB_CANCER = 0.0002

        self.ALIGNMENT_RADIUS_KILLER = 12.0
        self.ALIGNMENT_RADIUS_CANCER = 10.0

    def _initialize_phenotypes(self):
        prob_sum = self.scout_prob + self.messenger_prob + self.killer_prob
        if not math.isclose(prob_sum, 1.0, abs_tol=1e-3):
            self.killer_prob = max(0.05, 1.0 - (self.scout_prob + self.messenger_prob))

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
        p = self.params
        dev = self.device
        gen = self.gen
        N = self.num_cells

        imm_scout = self.is_immune & (self.phenotype == 0)
        imm_msg = self.is_immune & (self.phenotype == 1)
        imm_killer = self.is_immune & (self.phenotype == 2)
        can_sessile = self.is_cancer & (self.phenotype == 0)
        can_evasive = self.is_cancer & (self.phenotype == 1)

        raw_speeds = torch.normal(mean=self.IMMUNE_BASE_MEAN, std=self.IMMUNE_BASE_STD,
                                  size=(N,), generator=gen, device=dev).clamp(min=0.5)

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

        # Steering persistence (Tau)
        tau_scout_lo = p.get("tau_scout_min", 0.840092)
        tau_scout_hi = p.get("tau_scout_max", 2.473364)
        tau_msg_lo = p.get("tau_msg_min", 2.447561)
        tau_msg_hi = p.get("tau_msg_max", 3.952755)
        tau_killer_lo = p.get("tau_killer_min", 3.204229)
        tau_killer_hi = p.get("tau_killer_max", 5.559914)
        tau_can_lo = p.get("tau_cancer_min", 2.612264)
        tau_can_hi = p.get("tau_cancer_max", 3.699117)

        self.tau = self._blend(imm_scout, tau_scout_lo, tau_scout_hi, z.clone())
        self.tau = self._blend(imm_msg, tau_msg_lo, tau_msg_hi, self.tau)
        self.tau = self._blend(imm_killer, tau_killer_lo, tau_killer_hi, self.tau)
        self.tau = torch.where(self.is_cancer, _sample_uniform(tau_can_lo, tau_can_hi, N, gen, dev), self.tau)

        # Directional noise scales
        n_scout_lo = p.get("noise_scout_min", 0.188765)
        n_scout_hi = p.get("noise_scout_max", 0.433819)
        n_msg_lo = p.get("noise_msg_min", 0.155717)
        n_msg_hi = p.get("noise_msg_max", 0.425253)
        n_killer_lo = p.get("noise_killer_min", 0.049216)
        n_killer_hi = p.get("noise_killer_max", 0.159468)
        n_can_sess_lo = p.get("noise_can_sess_min", 0.043373)
        n_can_sess_hi = p.get("noise_can_sess_max", 0.138672)
        n_can_evas_lo = p.get("noise_can_evas_min", 0.092052)
        n_can_evas_hi = p.get("noise_can_evas_max", 0.292168)

        self.noise_scale = self._blend(imm_scout, n_scout_lo, n_scout_hi, z.clone())
        self.noise_scale = self._blend(imm_msg, n_msg_lo, n_msg_hi, self.noise_scale)
        self.noise_scale = self._blend(imm_killer, n_killer_lo, n_killer_hi, self.noise_scale)
        self.noise_scale = self._blend(can_sessile, n_can_sess_lo, n_can_sess_hi, self.noise_scale)
        self.noise_scale = self._blend(can_evasive, n_can_evas_lo, n_can_evas_hi, self.noise_scale)

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
        # Cancer IDs start at 1001, or after the last immune ID for >1000 immune cells
        cancer_start = max(1000, self.num_immune) + 1
        cancer_ids = torch.arange(cancer_start, cancer_start + self.num_cancer, dtype=torch.long, device=dev)
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

        # GPU Recording Buffers
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
        
        self.rec_dx_acc = torch.zeros((N, T), device=dev)
        self.rec_dy_acc = torch.zeros((N, T), device=dev)
        self.rec_dt_acc = torch.zeros((N, T), device=dev)

        self.initial_pos = self.positions.clone()
        self.cum_dx_acc = torch.zeros(N, device=dev)
        self.cum_dy_acc = torch.zeros(N, device=dev)
        self.cum_dist_traveled = torch.zeros(N, device=dev)

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
    # KINEMATIC RECORDER (ACCUMULATED SPATIAL & TEMPORAL METRICS)
    # ============================================================
    def _write_frame(self, t):
        self.stats_immune_alive[t] = int((self.is_immune & self.active_mask).sum().item())
        self.stats_cancer_alive[t] = int((self.is_cancer & self.active_mask).sum().item())

        self.recorded_active_mask[:, t] = self.active_mask

        curr_pos_x = self.positions[:, 0]
        curr_pos_y = self.positions[:, 1]

        self.rec_pos_x[:, t] = curr_pos_x
        self.rec_pos_y[:, t] = curr_pos_y

        dt_sec = self.dt * 10.0  # 10.0 seconds per step
        elapsed_seconds = float(t * dt_sec)

        if t == 0:
            # Frame 0 is the first recorded position, so it is the track origin.
            self.initial_pos = self.positions.clone()

            self.rec_dx_prev[:, 0] = float('nan')
            self.rec_dy_prev[:, 0] = float('nan')
            self.rec_disp_prev[:, 0] = float('nan')
            self.rec_vel_x[:, 0] = float('nan')
            self.rec_vel_y[:, 0] = float('nan')
            self.rec_speed[:, 0] = float('nan')
            
            self.rec_dx_acc[:, 0] = 0.0
            self.rec_dy_acc[:, 0] = 0.0
            self.rec_dt_acc[:, 0] = 0.0

            self.rec_dist_traveled[:, 0] = 0.0
            self.rec_dx_orig[:, 0] = 0.0
            self.rec_dy_orig[:, 0] = 0.0
            self.rec_disp_orig[:, 0] = 0.0

            # Undefined before any movement; 0.0 matches the experimental tracking data.
            self.rec_path_efficiency[:, 0] = 0.0
            self.rec_avg_speed[:, 0] = 0.0
        else:
            prev_x = self.rec_pos_x[:, t - 1]
            prev_y = self.rec_pos_y[:, t - 1]

            # 1. Step displacements
            dx_prev = curr_pos_x - prev_x
            dy_prev = curr_pos_y - prev_y
            disp_prev = torch.sqrt(dx_prev**2 + dy_prev**2)

            # 2. Instantaneous velocities and speed (µm/s)
            vel_x_inst = dx_prev / dt_sec
            vel_y_inst = dy_prev / dt_sec
            speed_inst = torch.sqrt(vel_x_inst**2 + vel_y_inst**2)

            # 3. Accumulated Distances (DX_ACC, DY_ACC in µm)
            self.cum_dx_acc += torch.abs(dx_prev)
            self.cum_dy_acc += torch.abs(dy_prev)

            # 4. Total Distance Traveled (µm): cumulative sum of step lengths since frame 0
            self.cum_dist_traveled += disp_prev
            dist_traveled = self.cum_dist_traveled.clone()

            # 5. Net Displacement from Origin (µm)
            orig_x = self.initial_pos[:, 0]
            orig_y = self.initial_pos[:, 1]
            dx_orig = curr_pos_x - orig_x
            dy_orig = curr_pos_y - orig_y
            disp_orig = torch.sqrt(dx_orig**2 + dy_orig**2)

            # 6. Path Efficiency (displacement can never exceed path length, so the
            #    clamp only guards against float32 rounding)
            path_eff = torch.where(
                dist_traveled > 1e-6,
                disp_orig / dist_traveled,
                torch.zeros_like(disp_orig)
            )
            path_eff = torch.clamp(path_eff, 0.0, 1.0)

            # 7. Average Speed (µm/s)
            avg_speed_val = dist_traveled / elapsed_seconds

            self.rec_dx_prev[:, t] = dx_prev
            self.rec_dy_prev[:, t] = dy_prev
            self.rec_disp_prev[:, t] = disp_prev
            self.rec_vel_x[:, t] = vel_x_inst
            self.rec_vel_y[:, t] = vel_y_inst
            self.rec_speed[:, t] = speed_inst

            self.rec_dx_acc[:, t] = self.cum_dx_acc
            self.rec_dy_acc[:, t] = self.cum_dy_acc
            self.rec_dt_acc[:, t] = elapsed_seconds

            self.rec_dx_orig[:, t] = dx_orig
            self.rec_dy_orig[:, t] = dy_orig
            self.rec_disp_orig[:, t] = disp_orig

            self.rec_dist_traveled[:, t] = dist_traveled
            self.rec_path_efficiency[:, t] = path_eff
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

        stats = {
            "immune_alive": self.stats_immune_alive,
            "cancer_alive": self.stats_cancer_alive,
            "kills_phase1": self.stats_kills_phase1,
            "counterkills_phase2": self.stats_counterkills_phase2,
        }
        return stats

    def export_kinematics_csv(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        T = self.timesteps

        track_ids = self.track_id.cpu().numpy()
        is_immune = self.is_immune.cpu().numpy()
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
        dx_acc = self.rec_dx_acc.cpu().numpy()
        dy_acc = self.rec_dy_acc.cpu().numpy()
        dt_acc = self.rec_dt_acc.cpu().numpy()

        t_cell_rows = []
        cancer_cell_rows = []

        for cell_idx in range(self.num_cells):
            t_id = track_ids[cell_idx]
            is_immune_cell = bool(is_immune[cell_idx])

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
                    dx_acc[cell_idx, t_idx],
                    dy_acc[cell_idx, t_idx],
                    dt_acc[cell_idx, t_idx],
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
            "SPEED", "AVERAGE_SPEED", "DX_ACC", "DY_ACC", "DT_ACC"
        ]

        df_tcell = pd.DataFrame(t_cell_rows, columns=columns)
        df_cancer = pd.DataFrame(cancer_cell_rows, columns=columns)

        tcell_filename = os.path.join(output_dir, f"{self.mode}_T-cell_kinematics.csv")
        cancer_filename = os.path.join(output_dir, f"{self.mode}_Cancer-cell_kinematics.csv")

        df_tcell.to_csv(tcell_filename, index=False)
        df_cancer.to_csv(cancer_filename, index=False)

        return tcell_filename, cancer_filename


# =============================================================================
# DUAL REGIME JSON RUNNER
# =============================================================================
def load_json_params(filepath):
    if not os.path.exists(filepath):
        print(f"[Warning] JSON file '{filepath}' not found. Falling back to default values.")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Loaded] Successfully ingested parameters from: {filepath}")
    return data


def run_dual_regime_simulations(
    killing_json_path="optimal_parameters_killing_0to5hrs.json",
    non_killing_json_path="optimal_parameters_non-killing_0to5hrs.json",
    timesteps=8635,
    output_dir="new_simulator/kinematic_csv_outputs",
    num_immune=1000,
    num_cancer=1000
):
    print("=" * 80)
    print("LAUNCHING DUAL REGIME SIMULATIONS FROM OPTIMIZED JSON ARTIFACTS")
    print(f"Horizon: {timesteps} frames (~24 hours) | Output: {output_dir}")
    print("=" * 80)

    # Ingest JSON parameter sets
    kill_params = load_json_params(killing_json_path)
    nonkill_params = load_json_params(non_killing_json_path)

    # 1. Execute Killing Simulation
    print("\n" + "-" * 80)
    print(">>> RUNNING KILLING REGIME SIMULATION")
    print("-" * 80)
    sim_kill = CellSimulation(
        num_immune=num_immune,
        num_cancer=num_cancer,
        timesteps=timesteps,
        mode="killing",
        seed=42,
        params_dict=kill_params
    )
    sim_kill.run_simulation()
    t_kill_csv, c_kill_csv = sim_kill.export_kinematics_csv(output_dir)
    print(f"Exported Killing Datasets:\n  - {t_kill_csv}\n  - {c_kill_csv}")

    # 2. Execute Non-Killing Simulation
    print("\n" + "-" * 80)
    print(">>> RUNNING NON-KILLING REGIME SIMULATION")
    print("-" * 80)
    sim_nonkill = CellSimulation(
        num_immune=num_immune,
        num_cancer=num_cancer,
        timesteps=timesteps,
        mode="non-killing",
        seed=142,
        params_dict=nonkill_params
    )
    sim_nonkill.run_simulation()
    t_nonkill_csv, c_nonkill_csv = sim_nonkill.export_kinematics_csv(output_dir)
    print(f"Exported Non-Killing Datasets:\n  - {t_nonkill_csv}\n  - {c_nonkill_csv}")

    print("\n" + "=" * 80)
    print("DUAL-REGIME EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_dual_regime_simulations(
        killing_json_path="optimal_parameters_killing_0to5hrs.json",
        non_killing_json_path="optimal_parameters_non-killing_0to5hrs.json",
        timesteps=8635,
        output_dir="new_simulator/kinematic_csv_outputs"
    )