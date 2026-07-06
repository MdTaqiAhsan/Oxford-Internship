"""
cell_simulator.py  (v3 — PyTorch + uniform spatial-hash grid, GPU-scalable)
============================================================================

Rewrite of v2 to scale from ~20 cells to 1,000-10,000+ cells.

WHAT CHANGED, AT A GLANCE
--------------------------
- Every tensor lives on `device` (CUDA if available, else CPU). No numpy
  inside the simulation loop; numpy is only touched once, at the very end,
  to hand back the array your Hopfield pipeline expects.
- All O(N^2) pairwise distance matrices are replaced by a uniform spatial
  hash grid: each cell only compares itself against cells in its own and
  the 8 neighboring grid buckets, not the whole population.
- The per-cell Python `for` loop over immune/cancer cells in v2 is gone.
  Every behavioral rule (sensing, steering, evasion, exhaustion, killing,
  collision, boundary) is expressed as whole-tensor operations.

WHAT DID NOT CHANGE
--------------------
- Public interface: `CellSimulation(...)`, `sim.run_simulation()`,
  `generate_dataset(mode, run_id)` all work exactly as before.
- Output contract: `trajectory_data.shape == (num_cells, timesteps, 5)`,
  feature order `[x, y, vx, vy, type]`, with the same 0/1/2/3/4 type
  encoding introduced in v2 (dead / immune-scout / immune-cytotoxic /
  cancer-sessile / cancer-evasive).
- Every named behavior from your list: scouts, cytotoxic cells, sessile
  and evasive cancer, probabilistic sensing, target locking, persistence
  (via the tau time-constant), acceleration/inertia, environmental noise,
  exhaustion/recovery, probabilistic + duration-gated killing, collision
  avoidance, and boundary repulsion are all still here.

ONE DELIBERATE BEHAVIORAL FIX (flagged, not hidden)
----------------------------------------------------
`width`/`height` now auto-scale to preserve the ORIGINAL cell density
(20 cells / 100x100 arena) unless you pass them explicitly. Without this,
2000 cells at the old fixed 100x100 size would be packed ~100x denser
than the simulation was ever tuned for -- everyone permanently colliding,
and every grid bucket containing a large fraction of all cells (which
also breaks the efficiency the grid is supposed to buy you). Pass
`width=..., height=...` explicitly to opt out of auto-scaling.

See the accompanying message for the full complexity/optimization writeup.
"""

import os
import math
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3x3 neighborhood offsets. Correct/sufficient as long as grid cell_size
# >= every interaction radius used anywhere in the simulation (guaranteed
# in __init__ below).
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
                 target_density=0.002,   # cells per unit^2, matches v1/v2's 20/(100*100)
                 max_per_cell=32,        # grid bucket capacity; see writeup for sizing
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

        # ---- auto-scale domain to preserve density unless overridden ----
        if width is None or height is None:
            side = math.sqrt(N / target_density)
            width = width if width is not None else side
            height = height if height is not None else side
        self.width = float(width)
        self.height = float(height)

        dev = self.device
        gen = self.gen

        # ---------------------------------------------------------------
        # Base class (fixed) and phenotype (drawn once per cell at init)
        # ---------------------------------------------------------------
        self.base_class = torch.cat([
            torch.ones(num_immune, dtype=torch.long, device=dev),
            2 * torch.ones(num_cancer, dtype=torch.long, device=dev),
        ])
        self.is_immune = self.base_class == 1
        self.is_cancer = self.base_class == 2

        phen_immune = (torch.rand(num_immune, generator=gen, device=dev) < 0.5).long()
        phen_cancer = (torch.rand(num_cancer, generator=gen, device=dev) < 0.4).long()  # p=0.4 evasive
        self.phenotype = torch.cat([phen_immune, phen_cancer])

        imm_scout = self.is_immune & (self.phenotype == 0)
        imm_cyto = self.is_immune & (self.phenotype == 1)
        can_sessile = self.is_cancer & (self.phenotype == 0)
        can_evasive = self.is_cancer & (self.phenotype == 1)

        def blend(mask, lo, hi, base):
            return torch.where(mask, _sample_uniform(lo, hi, N, gen, dev), base)

        z = torch.zeros(N, device=dev)
        self.max_speed = blend(imm_scout, 1.6, 2.0, z)
        self.max_speed = blend(imm_cyto, 1.0, 1.4, self.max_speed)
        self.max_speed = blend(can_sessile, 0.15, 0.35, self.max_speed)
        self.max_speed = blend(can_evasive, 0.8, 1.3, self.max_speed)

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

        self.tau = blend(imm_scout, 1.5, 2.5, z.clone())
        self.tau = blend(imm_cyto, 3.0, 5.0, self.tau)
        self.tau = torch.where(self.is_cancer, _sample_uniform(2.0, 4.0, N, gen, dev), self.tau)

        self.noise_scale = blend(imm_scout, 0.15, 0.25, z.clone())
        self.noise_scale = blend(imm_cyto, 0.05, 0.12, self.noise_scale)
        self.noise_scale = blend(can_sessile, 0.05, 0.10, self.noise_scale)
        self.noise_scale = blend(can_evasive, 0.10, 0.18, self.noise_scale)

        # ---------------------------------------------------------------
        # Dynamic state
        # ---------------------------------------------------------------
        self.positions = torch.stack([
            _sample_uniform(0, self.width, N, gen, dev),
            _sample_uniform(0, self.height, N, gen, dev),
        ], dim=1)
        self.velocities = torch.stack([
            _sample_uniform(-0.3, 0.3, N, gen, dev),
            _sample_uniform(-0.3, 0.3, N, gen, dev),
        ], dim=1)
        self.active_mask = torch.ones(N, dtype=torch.bool, device=dev)
        self.energy = torch.ones(N, device=dev)
        self.contact_timer = torch.zeros(N, device=dev)
        self.locked_target = torch.full((N,), -1, dtype=torch.long, device=dev)

        self.KILL_RADIUS = 2.5
        self.ENGAGE_STEPS_REQUIRED = 4
        self.MIN_SEPARATION = 1.5
        self.BOUNDARY_MARGIN = 8.0

        # ---------------------------------------------------------------
        # Spatial grid setup. cell_size MUST be >= the largest interaction
        # radius used anywhere (sensing_radius*1.5 is the widest one, for
        # the fallback chemotaxis centroid) so the 3x3 neighborhood is
        # guaranteed to be a superset of the true neighbor set.
        # ---------------------------------------------------------------
        widest_radius = float((self.sensing_radius * 1.5).max().item())
        self.cell_size = max(widest_radius, self.MIN_SEPARATION) + 1e-3
        self.grid_w = max(1, int(math.ceil(self.width / self.cell_size)))
        self.grid_h = max(1, int(math.ceil(self.height / self.cell_size)))
        self.num_grid_cells = self.grid_w * self.grid_h
        self.max_per_cell = max_per_cell

        # trajectory buffer lives on-device for the whole run; only moved
        # to host memory once, at the end of run_simulation().
        self.trajectory_data = torch.zeros((N, timesteps, 5), device=dev)

        # reusable accel buffer (avoids a fresh allocation every step)
        self._accel = torch.zeros((N, 2), device=dev)

    # =====================================================================
    # Spatial hash grid: build + neighbor gather
    # =====================================================================
    def build_spatial_grid(self):
        """Bucket every active cell into a uniform grid cell via a counting
        sort. Returns a (num_grid_cells+1, max_per_cell) table of particle
        indices (-1 = empty slot); the extra +1 bucket is a dump for
        inactive (dead) cells so they're never returned as neighbors."""
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

        if self.debug:
            overflow = int((rank_in_cell >= self.max_per_cell).sum().item())
            if overflow > 0:
                print(f"[cell_simulator] WARNING: {overflow} cell(s) overflowed "
                      f"max_per_cell={self.max_per_cell}; increase it or shrink cell_size.")

        return table, gx, gy

    def find_neighbors(self):
        """For every cell, gather candidate neighbor indices from its own
        grid cell and the 8 adjacent ones. Returns:
            candidates : (N, 9*max_per_cell) long   -- particle indices, -1 padded
            valid      : (N, 9*max_per_cell) bool
            dist       : (N, 9*max_per_cell) float   -- distance to each candidate
                         (inf where invalid or self)
            diff       : (N, 9*max_per_cell, 2) float -- candidate_pos - self_pos
        """
        table, gx, gy = self.build_spatial_grid()
        chunks = []
        for dx, dy in _NEIGHBOR_OFFSETS:
            ngx = torch.clamp(gx + dx, 0, self.grid_w - 1)
            ngy = torch.clamp(gy + dy, 0, self.grid_h - 1)
            ncell = ngy * self.grid_w + ngx
            chunks.append(table[ncell])
        candidates = torch.cat(chunks, dim=1)               # (N, 9*max_per_cell)
        valid = candidates >= 0

        cand_pos = self.positions[candidates.clamp(min=0)]   # (N,K,2)
        diff = cand_pos - self.positions.unsqueeze(1)         # (N,K,2)
        dist = torch.norm(diff, dim=2)

        self_mask = candidates == torch.arange(self.num_cells, device=self.device).unsqueeze(1)
        invalid = (~valid) | self_mask
        dist = torch.where(invalid, torch.full_like(dist, float('inf')), dist)

        return candidates, valid, dist, diff

    # =====================================================================
    # Behavioral update functions (each modifies self._accel in place)
    # =====================================================================
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

        # fallback: weak chemotaxis toward centroid of widely-sensed cancer cells
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

        drift = torch.randn(self.num_cells, 2, generator=self.gen, device=self.device) * 0.08 \
            - 0.1 * self.velocities

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
        unit_away = -diff / (dist.unsqueeze(2) + 1e-6)   # push away from the overlapping candidate
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

    # =====================================================================
    # Orchestration
    # =====================================================================
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
        # single host sync at the very end, not per-step
        return self.trajectory_data.cpu().numpy()


# ==========================================
# DATASET GENERATION (unchanged interface)
# ==========================================
def generate_dataset(mode, run_id, num_immune=1000, num_cancer=1000, timesteps=120,
                      output_dir="new_simulator/new_simulation_data", **kwargs):
    """Runs one simulation and saves it exactly as before, so downstream
    code (data_dir + data_{mode}_{i}.npy convention) needs zero changes."""
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
    for i in range(50):
        generate_dataset('killing', i)
        generate_dataset('non-killing', i)
    print("All 100 simulation files successfully generated!")