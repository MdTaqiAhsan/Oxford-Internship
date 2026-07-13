"""
new_cell_simulator.py  (v3.4 — PyTorch + uniform spatial-hash grid, GPU-scalable)
============================================================================
Revised simulation engine featuring mode-dependent experimental immune velocity 
sampling, polarized initial velocity mapping, and explicit physical unit constraints.

PHYSICAL UNIT CONVENTIONS & ASSUMPTIONS:
----------------------------------------
- Spatial Coordinates (x, y, radii) : Expressed in micrometers (μm).
- Temporal Step (dt = 1.0)           : Represents exactly 1 minute of real time.
- Velocities & Speeds (vx, vy, max) : Expressed in μm/min.
- Cell Density                       : Standardized to cells per μm^2.
"""

import os
import math
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),  (0, 0),  (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]


def _sample_uniform(lo, hi, n, generator, device):
    return lo + (hi - lo) * torch.rand(n, generator=generator, device=device)


class CellSimulation:
    def __init__(self,
                 num_immune=8, num_cancer=12,
                 width=None, height=None,
                 timesteps=120,
                 mode='killing',
                 dt=1.0, 
                 seed=None,
                 target_density=0.002,   
                 max_per_cell=32,        
                 device=None,
                 debug=False):
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
        self.mode = mode
        self.dt = dt
        self.debug = debug

        if width is None or height is None:
            side = math.sqrt(N / target_density)
            width = width if width is not None else side
            height = height if height is not None else side
        self.width = float(width)     # Dimension in μm
        self.height = float(height)   # Dimension in μm

        dev = self.device
        gen = self.gen

        # ---------------------------------------------------------------
        # Base class and phenotype tracking
        # ---------------------------------------------------------------
        self.base_class = torch.cat([
            torch.ones(num_immune, dtype=torch.long, device=dev),
            2 * torch.ones(num_cancer, dtype=torch.long, device=dev),
        ])
        self.is_immune = self.base_class == 1
        self.is_cancer = self.base_class == 2

        phen_immune = (torch.rand(num_immune, generator=gen, device=dev) < 0.5).long()
        phen_cancer = (torch.rand(num_cancer, generator=gen, device=dev) < 0.4).long()  
        self.phenotype = torch.cat([phen_immune, phen_cancer])

        imm_scout = self.is_immune & (self.phenotype == 0)
        imm_cyto = self.is_immune & (self.phenotype == 1)
        can_sessile = self.is_cancer & (self.phenotype == 0)
        can_evasive = self.is_cancer & (self.phenotype == 1)

        def blend(mask, lo, hi, base):
            return torch.where(mask, _sample_uniform(lo, hi, N, gen, dev), base)

        # ---------------------------------------------------------------
        # MODIFICATION: MODE-DEPENDENT EXPERIMENTAL IMMUNE SPEED SAMPLING
        # ---------------------------------------------------------------
        # Select target Gaussian hyper-parameters dynamically based on simulation mode boundary rules
        if self.mode == 'killing':
            target_mean = 5.0
            target_std = 1.6
        else:  # 'non-killing'
            target_mean = 8.5
            target_std = 2.0

        raw_speeds = torch.normal(mean=target_mean, std=target_std, size=(N,), generator=gen, device=dev)
        
        # In-place iterative positive sampling guard to preserve valid tail parameters on GPU
        neg_mask = raw_speeds <= 0.0
        while neg_mask.any():
            num_bad = int(neg_mask.sum().item())
            replacements = torch.normal(mean=target_mean, std=target_std, size=(num_bad,), generator=gen, device=dev)
            raw_speeds[neg_mask] = replacements
            neg_mask = raw_speeds <= 0.0

        z = torch.zeros(N, device=dev)
        self.max_speed = torch.where(self.is_immune, raw_speeds, z)

        # ---------------------------------------------------------------
        # UNCHANGED: CANCER MAX SPEEDS 
        # ---------------------------------------------------------------
        self.max_speed = blend(can_sessile, 0.5, 1.5, self.max_speed)  # Speed in μm/min
        self.max_speed = blend(can_evasive, 2.0, 4.0, self.max_speed)  # Speed in μm/min

        # Phenomenological behavioral parameter configurations (Sensing in μm)
        self.sensing_radius = blend(imm_scout, 18, 24, z.clone())
        self.sensing_radius = blend(imm_cyto, 10, 16, self.sensing_radius)
        self.sensing_radius = torch.where(can_evasive | can_sessile,
                                           _sample_uniform(12, 18, N, gen, dev),
                                           self.sensing_radius)

        self.sensing_prob = blend(imm_scout, 0.5, 0.7, z.clone())
        self.sensing_prob = blend(imm_cyto, 0.7, 0.9, self.sensing_prob)
        self.sensing_prob = torch.where(self.is_cancer, torch.ones(N, device=dev), self.sensing_prob)

        self.kill_rate = blend(imm_scout, 0.05, 0.10, z.clone())
        self.kill_rate = blend(imm_cyto, 0.20, 0.35, self.kill_rate)
        if mode == 'non-killing':
            self.kill_rate = torch.where(self.is_immune, self.kill_rate * 0.03, self.kill_rate)

        # Persistence time-constants (tau in minutes)
        self.tau = blend(imm_scout, 1.5, 2.5, z.clone())
        self.tau = blend(imm_cyto, 3.0, 5.0, self.tau)
        self.tau = torch.where(self.is_cancer, _sample_uniform(2.0, 4.0, N, gen, dev), self.tau)

        # Re-calibrated Extrinsic Noise Coefficients
        self.noise_scale = blend(imm_scout, 0.15, 0.25, z.clone())
        self.noise_scale = blend(imm_cyto, 0.05, 0.12, self.noise_scale)
        self.noise_scale = blend(can_sessile, 0.05, 0.15, self.noise_scale)
        self.noise_scale = blend(can_evasive, 0.10, 0.30, self.noise_scale)

        # Dynamic State Variables (Positions in μm)
        self.positions = torch.stack([
            _sample_uniform(0, self.width, N, gen, dev),
            _sample_uniform(0, self.height, N, gen, dev),
        ], dim=1)
        
        # Polar Initial Velocity Assignment
        init_speeds = torch.rand(N, generator=gen, device=dev) * self.max_speed
        init_angles = torch.rand(N, generator=gen, device=dev) * (2.0 * math.pi)
        
        self.velocities = torch.stack([
            init_speeds * torch.cos(init_angles),
            init_speeds * torch.sin(init_angles)
        ], dim=1)

        self.active_mask = torch.ones(N, dtype=torch.bool, device=dev)
        self.energy = torch.ones(N, device=dev)
        self.contact_timer = torch.zeros(N, device=dev)
        self.locked_target = torch.full((N,), -1, dtype=torch.long, device=dev)

        self.KILL_RADIUS = 2.5            # Distance bound in μm
        self.ENGAGE_STEPS_REQUIRED = 4    # Timestep iterations (4 minutes)
        self.MIN_SEPARATION = 1.5         # Collision threshold in μm
        self.BOUNDARY_MARGIN = 8.0         # Boundary threshold in μm

        widest_radius = float((self.sensing_radius * 1.5).max().item())
        self.cell_size = max(widest_radius, self.MIN_SEPARATION) + 1e-3
        self.grid_w = max(1, int(math.ceil(self.width / self.cell_size)))
        self.grid_h = max(1, int(math.ceil(self.height / self.cell_size)))
        self.num_grid_cells = self.grid_w * self.grid_h
        self.max_per_cell = max_per_cell

        self.trajectory_data = torch.zeros((N, timesteps, 5), device=dev)
        self._accel = torch.zeros((N, 2), device=dev)

    def build_spatial_grid(self):
        gx = torch.clamp((self.positions[:, 0] / self.cell_size).long(), 0, self.grid_w - 1)
        gy = torch.clamp((self.positions[:, 1] / self.cell_size).long(), 0, self.grid_h - 1)
        cell_id = gy * self.grid_w + gx
        dummy_id = self.num_grid_cells
        cell_id = torch.where(self.active_mask, cell_id, torch.full_like(cell_id, dummy_id))

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

    def find_neighbors(self):
        table, gx, gy = self.build_spatial_grid()
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
        invalid = (~valid) | self_mask
        dist = torch.where(invalid, torch.full_like(dist, float('inf')), dist)

        return candidates, valid, dist, diff

    def update_immune_cells(self, candidates, dist, diff):
        cand_is_cancer = self.is_cancer[candidates.clamp(min=0)]
        dist_cancer = torch.where(cand_is_cancer, dist, torch.full_like(dist, float('inf')))

        within_range = dist_cancer <= self.sensing_radius.unsqueeze(1)
        nearest_val, nearest_idx = torch.min(dist_cancer, dim=1)
        has_any_within = within_range.any(dim=1)
        nearest_particle = torch.gather(candidates, 1, nearest_idx.unsqueeze(1)).squeeze(1)

        detect_roll = torch.rand(self.num_cells, generator=self.gen, device=self.device) < self.sensing_prob

        tgt = self.locked_target
        tgt_alive = (tgt >= 0) & self.active_mask[tgt.clamp(min=0)]
        tgt = torch.where(tgt_alive, tgt, torch.full_like(tgt, -1))

        acquire = has_any_within & detect_roll & self.is_immune & self.active_mask
        tgt = torch.where(acquire, nearest_particle, tgt)
        self.locked_target = tgt
        has_target = (tgt >= 0) & self.is_immune & self.active_mask

        target_pos = self.positions[tgt.clamp(min=0)]
        direction = target_pos - self.positions
        d = torch.norm(direction, dim=1) + 1e-6
        desired_speed = self.max_speed * (0.4 + 0.6 * self.energy)
        desired_vel_target = direction / d.unsqueeze(1) * desired_speed.unsqueeze(1)
        engaged = has_target & (d <= self.sensing_radius)

        wide_range = dist_cancer <= (self.sensing_radius * 1.5).unsqueeze(1)
        any_wide = wide_range.any(dim=1)
        mask_f = (wide_range & cand_is_cancer).float().unsqueeze(2)
        cand_pos = self.positions[candidates.clamp(min=0)]
        sum_pos = (cand_pos * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1e-6)
        centroid = sum_pos / count
        dir_c = centroid - self.positions
        dcn = torch.norm(dir_c, dim=1) + 1e-6
        desired_vel_centroid = dir_c / dcn.unsqueeze(1) * (self.max_speed * 0.5).unsqueeze(1)
        random_explore = torch.randn(self.num_cells, 2, generator=self.gen, device=self.device) \
            * (self.max_speed * 0.3).unsqueeze(1)
        desired_vel_fallback = torch.where(any_wide.unsqueeze(1), desired_vel_centroid, random_explore)

        desired_vel = torch.where(has_target.unsqueeze(1), desired_vel_target, desired_vel_fallback)
        steer_accel = (desired_vel - self.velocities) / self.tau.unsqueeze(1)

        apply_mask = (self.is_immune & self.active_mask).unsqueeze(1)
        self._accel += torch.where(apply_mask, steer_accel, torch.zeros_like(steer_accel))

        active_immune = self.is_immune & self.active_mask
        energy_delta = torch.where(engaged, torch.full_like(self.energy, -0.01),
                                    torch.full_like(self.energy, 0.02))
        self.energy = torch.where(active_immune,
                                   torch.clamp(self.energy + energy_delta, 0.05, 1.0),
                                   self.energy)

    def update_cancer_cells(self, candidates, dist, diff):
        is_evasive = self.is_cancer & (self.phenotype == 1)
        cand_is_immune = self.is_immune[candidates.clamp(min=0)]
        dist_immune = torch.where(cand_is_immune, dist, torch.full_like(dist, float('inf')))

        within = dist_immune <= self.sensing_radius.unsqueeze(1)
        nearest_idx = torch.argmin(dist_immune, dim=1)
        has_threat = within.any(dim=1) & is_evasive & self.active_mask

        flee_dir = -torch.gather(diff, 1, nearest_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        dnorm = torch.norm(flee_dir, dim=1) + 1e-6
        desired_vel_flee = flee_dir / dnorm.unsqueeze(1) * self.max_speed.unsqueeze(1)
        steer_flee = (desired_vel_flee - self.velocities) / self.tau.unsqueeze(1)

        drift = torch.randn(self.num_cells, 2, generator=self.gen, device=self.device) * 0.05 \
            - 0.10 * self.velocities

        cancer_accel = torch.where(has_threat.unsqueeze(1), steer_flee, drift)
        apply_mask = (self.is_cancer & self.active_mask).unsqueeze(1)
        self._accel += torch.where(apply_mask, cancer_accel, torch.zeros_like(cancer_accel))

    def apply_environment_noise(self):
        noise = torch.randn(self.num_cells, 2, generator=self.gen, device=self.device) \
            * self.noise_scale.unsqueeze(1)
        apply_mask = self.active_mask.unsqueeze(1)
        self._accel += torch.where(apply_mask, noise, torch.zeros_like(noise))

    def resolve_collisions(self, candidates, valid, dist, diff):
        close = valid & (dist < self.MIN_SEPARATION) \
            & self.active_mask.unsqueeze(1) & self.active_mask[candidates.clamp(min=0)]
        overlap = torch.clamp(self.MIN_SEPARATION - dist, min=0)
        unit_away = -diff / (dist.unsqueeze(2) + 1e-6)   
        repulse = (unit_away * overlap.unsqueeze(2) * close.unsqueeze(2)).sum(dim=1) * 0.5
        self._accel += repulse

    def apply_boundary_forces(self):
        x, y = self.positions[:, 0], self.positions[:, 1]
        push_x = torch.where(x < self.BOUNDARY_MARGIN, (self.BOUNDARY_MARGIN - x) * 0.3,
                              torch.zeros_like(x))
        push_x = torch.where(x > self.width - self.BOUNDARY_MARGIN,
                              -(x - (self.width - self.BOUNDARY_MARGIN)) * 0.3, push_x)
        push_y = torch.where(y < self.BOUNDARY_MARGIN, (self.BOUNDARY_MARGIN - y) * 0.3,
                              torch.zeros_like(y))
        push_y = torch.where(y > self.height - self.BOUNDARY_MARGIN,
                              -(y - (self.height - self.BOUNDARY_MARGIN)) * 0.3, push_y)
        boundary_accel = torch.stack([push_x, push_y], dim=1)
        apply_mask = self.active_mask.unsqueeze(1)
        self._accel += torch.where(apply_mask, boundary_accel, torch.zeros_like(boundary_accel))

    def integrate_motion(self):
        active = self.active_mask.unsqueeze(1)
        self.velocities = torch.where(active, self.velocities + self._accel * self.dt,
                                       torch.zeros_like(self.velocities))
        speed = torch.norm(self.velocities, dim=1) + 1e-9
        scale = torch.clamp(self.max_speed / speed, max=1.0)
        self.velocities = self.velocities * scale.unsqueeze(1)
        self.positions = torch.where(active, self.positions + self.velocities * self.dt, self.positions)
        self.positions[:, 0] = torch.clamp(self.positions[:, 0], 0, self.width)
        self.positions[:, 1] = torch.clamp(self.positions[:, 1], 0, self.height)

    def perform_killing(self, candidates, dist):
        cand_is_immune = self.is_immune[candidates.clamp(min=0)]
        dist_immune = torch.where(cand_is_immune, dist, torch.full_like(dist, float('inf')))
        nearest_val, nearest_idx = torch.min(dist_immune, dim=1)
        nearest_particle = torch.gather(candidates, 1, nearest_idx.unsqueeze(1)).squeeze(1)

        is_cancer_active = self.is_cancer & self.active_mask
        in_contact = is_cancer_active & (nearest_val < self.KILL_RADIUS)
        self.contact_timer = torch.where(in_contact, self.contact_timer + 1, torch.zeros_like(self.contact_timer))

        ready = in_contact & (self.contact_timer >= self.ENGAGE_STEPS_REQUIRED)
        attacker = nearest_particle.clamp(min=0)
        p_kill = self.kill_rate[attacker] * (0.5 + 0.5 * self.energy[attacker])
        roll = torch.rand(self.num_cells, generator=self.gen, device=self.device)
        killed_now = ready & (roll < p_kill)

        self.active_mask = self.active_mask & (~killed_now)
        self.contact_timer = torch.where(killed_now, torch.zeros_like(self.contact_timer), self.contact_timer)

        if killed_now.any():
            dead_ids = torch.nonzero(killed_now, as_tuple=False).squeeze(1)
            is_targeting_dead = torch.isin(self.locked_target, dead_ids)
            self.locked_target = torch.where(is_targeting_dead, torch.full_like(self.locked_target, -1),
                                              self.locked_target)
            attacker_ids = attacker[killed_now]
            self.energy[attacker_ids] = torch.clamp(self.energy[attacker_ids] + 0.15, max=1.0)

    def _type_column(self):
        type_col = torch.zeros(self.num_cells, device=self.device)
        immune_mask = self.is_immune & self.active_mask
        cancer_mask = self.is_cancer & self.active_mask
        type_col = torch.where(immune_mask, 1.0 + self.phenotype.float(), type_col)
        type_col = torch.where(cancer_mask, 3.0 + self.phenotype.float(), type_col)
        return type_col

    def _write_frame(self, t):
        self.trajectory_data[:, t, 0:2] = self.positions
        self.trajectory_data[:, t, 2:4] = self.velocities
        self.trajectory_data[:, t, 4] = self._type_column()

    def step(self, t):
        self._accel.zero_()
        candidates, valid, dist, diff = self.find_neighbors()

        self.update_immune_cells(candidates, dist, diff)
        self.update_cancer_cells(candidates, dist, diff)
        self.apply_environment_noise()
        self.resolve_collisions(candidates, valid, dist, diff)
        self.apply_boundary_forces()

        self.integrate_motion()
        self.perform_killing(candidates, dist)
        self._write_frame(t)

    @torch.no_grad()
    def run_simulation(self):
        for t in range(self.timesteps):
            self.step(t)
        return self.trajectory_data.cpu().numpy()


# ==========================================
# DATASET GENERATION
# ==========================================
def generate_dataset(mode, run_id, num_immune=10000, num_cancer=10000, timesteps=120,
                      output_dir="new_simulator/new_simulation_data", **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seed = hash((mode, run_id)) % (2**32)
    sim = CellSimulation(num_immune=num_immune, num_cancer=num_cancer,
                          timesteps=timesteps, mode=mode, seed=seed, **kwargs)
    trajectories = sim.run_simulation()

    filename = os.path.join(output_dir, f"data_{mode}_{run_id}.npy")
    np.save(filename, trajectories)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("Generating 50 Killing and 50 Non-Killing simulations inside 'new_simulator/new_simulation_data/'...")
    for i in range(50,51):
        generate_dataset('killing', i)
        generate_dataset('non-killing', i)
    print("All 100 simulation files successfully generated!")