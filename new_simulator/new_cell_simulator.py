"""
cell_simulator.py  (v2 — biologically richer, still Hopfield-pipeline-compatible)
=================================================================================

DROP-IN REPLACEMENT for the original simulator.

Output contract (UNCHANGED, so hopfield_timeframe.ipynb needs no edits):
    trajectory_data.shape == (num_cells, timesteps, 5)
    feature order          == [x, y, vx, vy, type]

The ONE semantic change: `type` now carries 5 possible values instead of 3,
so a single scalar channel can carry phenotype information:
    0.0 -> dead / inactive
    1.0 -> immune, phenotype "scout"      (fast, low persistence, weaker kill)
    2.0 -> immune, phenotype "cytotoxic"  (slower, high persistence, strong kill)
    3.0 -> cancer, phenotype "sessile"    (low motility, doesn't evade)
    4.0 -> cancer, phenotype "evasive"    (actively flees nearby immune cells)
This is safe for the existing training notebook because it only ever
z-score-normalizes `type` as a continuous feature; it never branches on it.

Everything below is vectorized with numpy (broadcasting / pairwise distance
matrices) rather than nested python double-loops, so it stays efficient even
though the behavioral model is much richer than v1.
"""

import numpy as np
import os


class CellSimulation:
    def __init__(self,
                 num_immune=8, num_cancer=12,
                 width=100, height=100, timesteps=120,
                 mode='killing',
                 dt=1.0,
                 seed=None):
        """
        mode: 'killing'      -> immune cells have normal cytotoxic efficacy
              'non-killing'  -> immune cells sense & pursue IDENTICALLY, but
                                 engagement rarely/never converts to a kill
                                 (models dysfunctional / non-cytotoxic immunity,
                                 NOT a different motion-generating process --
                                 this is the fix for the killing/non-killing
                                 motion confound of v1).
        seed: optional int for fully reproducible individual simulations.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.num_immune = num_immune
        self.num_cancer = num_cancer
        self.num_cells = num_immune + num_cancer
        self.width = width
        self.height = height
        self.timesteps = timesteps
        self.mode = mode
        self.dt = dt

        N = self.num_cells
        rng = self.rng

        # ---------------------------------------------------------------
        # Base class (fixed) and phenotype (drawn once per cell at init)
        # ---------------------------------------------------------------
        # base_class: 1 = immune, 2 = cancer  (internal bookkeeping only)
        self.base_class = np.array([1] * num_immune + [2] * num_cancer)

        # phenotype: for immune, 0="scout" / 1="cytotoxic"
        #            for cancer, 0="sessile" / 1="evasive"
        self.phenotype = np.zeros(N, dtype=int)
        self.phenotype[:num_immune] = rng.choice([0, 1], size=num_immune, p=[0.5, 0.5])
        self.phenotype[num_immune:] = rng.choice([0, 1], size=num_cancer, p=[0.6, 0.4])

        # ---------------------------------------------------------------
        # Per-cell physical / behavioral parameters (phenotype-conditioned)
        # ---------------------------------------------------------------
        self.max_speed = np.zeros(N)
        self.sensing_radius = np.zeros(N)
        self.sensing_prob = np.zeros(N)      # P(detect target this step | in range)
        self.kill_rate = np.zeros(N)         # P(kill this step | engaged long enough)
        self.tau = np.zeros(N)               # steering time-constant (persistence)
        self.noise_scale = np.zeros(N)       # environmental jitter magnitude

        for i in range(num_immune):
            if self.phenotype[i] == 0:  # scout: fast, low persistence, weak kill
                self.max_speed[i] = rng.uniform(1.6, 2.0)
                self.sensing_radius[i] = rng.uniform(18, 24)
                self.sensing_prob[i] = rng.uniform(0.5, 0.7)
                self.kill_rate[i] = rng.uniform(0.05, 0.10)
                self.tau[i] = rng.uniform(1.5, 2.5)      # steers quickly -> jittery
                self.noise_scale[i] = rng.uniform(0.15, 0.25)
            else:  # cytotoxic: slower, high persistence, strong kill
                self.max_speed[i] = rng.uniform(1.0, 1.4)
                self.sensing_radius[i] = rng.uniform(10, 16)
                self.sensing_prob[i] = rng.uniform(0.7, 0.9)
                self.kill_rate[i] = rng.uniform(0.20, 0.35)
                self.tau[i] = rng.uniform(3.0, 5.0)      # steers slowly -> persistent
                self.noise_scale[i] = rng.uniform(0.05, 0.12)

        for j in range(num_immune, N):
            ph = self.phenotype[j]
            if ph == 0:  # sessile
                self.max_speed[j] = rng.uniform(0.15, 0.35)
                self.noise_scale[j] = rng.uniform(0.05, 0.10)
            else:  # evasive
                self.max_speed[j] = rng.uniform(0.8, 1.3)
                self.noise_scale[j] = rng.uniform(0.10, 0.18)
            self.tau[j] = rng.uniform(2.0, 4.0)
            self.sensing_radius[j] = rng.uniform(12, 18)   # evasion detection range
            self.sensing_prob[j] = 1.0                     # cancer "feels" nearby threat directly

        # Non-killing mode: identical sensing/steering, engagement just rarely
        # converts to a kill (dysfunctional cytotoxicity), NOT a different
        # movement process. This is the deliberate fix for the v1 confound.
        if mode == 'non-killing':
            self.kill_rate[:num_immune] *= 0.03   # engagement happens, kills essentially don't

        # ---------------------------------------------------------------
        # Dynamic state
        # ---------------------------------------------------------------
        self.positions = rng.uniform(0, [width, height], size=(N, 2))
        self.velocities = rng.uniform(-0.3, 0.3, size=(N, 2))
        self.active_mask = np.ones(N, dtype=bool)   # alive?
        self.energy = np.ones(N)                    # immune exhaustion state, 1=fresh
        self.contact_timer = np.zeros(N)             # cancer-side: consecutive engaged steps
        self.locked_target = np.full(N, -1, dtype=int)  # immune-side: locked cancer idx (-1=none)

        self.KILL_RADIUS = 2.5           # must be this close to count as "in contact"
        self.ENGAGE_STEPS_REQUIRED = 4   # sustained contact before kill rolls begin
        self.MIN_SEPARATION = 1.5        # soft collision distance
        self.BOUNDARY_MARGIN = 8.0       # distance from wall where soft repulsion kicks in

        self.trajectory_data = np.zeros((N, timesteps, 5))

    # ---------------------------------------------------------------
    # Helper: encode current type column (0/1/2/3/4 scheme)
    # ---------------------------------------------------------------
    def _type_column(self):
        type_col = np.zeros(self.num_cells)
        immune_mask = (self.base_class == 1) & self.active_mask
        cancer_mask = (self.base_class == 2) & self.active_mask
        # immune: phenotype 0 -> 1.0, phenotype 1 -> 2.0
        type_col[immune_mask] = 1.0 + self.phenotype[immune_mask]
        # cancer: phenotype 0 -> 3.0, phenotype 1 -> 4.0
        type_col[cancer_mask] = 3.0 + self.phenotype[cancer_mask]
        # dead cells (any class) stay 0.0 by construction (active_mask False -> not set above)
        return type_col

    # ---------------------------------------------------------------
    # One simulation step (vectorized)
    # ---------------------------------------------------------------
    def step(self, t):
        N = self.num_cells
        rng = self.rng
        immune_idx = np.where((self.base_class == 1) & self.active_mask)[0]
        cancer_idx = np.where((self.base_class == 2) & self.active_mask)[0]

        accel = np.zeros((N, 2))

        # =============================================================
        # 1) IMMUNE behavior: probabilistic sensing + steering toward a
        #    locked target, chemotaxis-style fallback when no target.
        # =============================================================
        if len(immune_idx) > 0 and len(cancer_idx) > 0:
            imm_pos = self.positions[immune_idx]              # (Ni,2)
            can_pos = self.positions[cancer_idx]               # (Nc,2)
            # pairwise distances immune -> cancer, vectorized
            diff = imm_pos[:, None, :] - can_pos[None, :, :]   # (Ni,Nc,2)
            dist = np.linalg.norm(diff, axis=2)                 # (Ni,Nc)

            for k, i in enumerate(immune_idx):
                # does the locked target still exist / is it still alive?
                tgt = self.locked_target[i]
                tgt_alive = tgt != -1 and self.active_mask[tgt]
                if not tgt_alive:
                    self.locked_target[i] = -1

                within_range = dist[k] <= self.sensing_radius[i]
                detects_now = rng.random() < self.sensing_prob[i]

                if within_range.any() and detects_now:
                    # (re)acquire nearest in-range cancer cell
                    nearest_local = np.argmin(np.where(within_range, dist[k], np.inf))
                    self.locked_target[i] = cancer_idx[nearest_local]
                    tgt_alive = True

                if self.locked_target[i] != -1 and tgt_alive:
                    target_pos = self.positions[self.locked_target[i]]
                    direction = target_pos - self.positions[i]
                    d = np.linalg.norm(direction) + 1e-6
                    desired_vel = (direction / d) * self.max_speed[i] * (0.4 + 0.6 * self.energy[i])
                    engaged = d <= self.sensing_radius[i]
                else:
                    # no locked target: weak chemotaxis pull toward the population
                    # centroid of sensed (in-range) cancer cells, else gentle
                    # random exploration (search behavior)
                    in_range_mask = dist[k] <= self.sensing_radius[i] * 1.5
                    if in_range_mask.any():
                        centroid = can_pos[in_range_mask].mean(axis=0)
                        direction = centroid - self.positions[i]
                        d = np.linalg.norm(direction) + 1e-6
                        desired_vel = (direction / d) * self.max_speed[i] * 0.5
                    else:
                        desired_vel = rng.normal(0, self.max_speed[i] * 0.3, size=2)
                    engaged = False

                # steering: accelerate toward desired velocity (persistence via tau)
                accel[i] = (desired_vel - self.velocities[i]) / self.tau[i]

                # exhaustion dynamics: deplete while actively engaged, regen otherwise
                if engaged:
                    self.energy[i] = max(0.05, self.energy[i] - 0.01)
                else:
                    self.energy[i] = min(1.0, self.energy[i] + 0.02)
        else:
            # no cancer left (or no immune left): idle wander
            for i in immune_idx:
                accel[i] = rng.normal(0, 0.05, size=2) - 0.05 * self.velocities[i]
                self.energy[i] = min(1.0, self.energy[i] + 0.02)

        # =============================================================
        # 2) CANCER behavior: sessile drift, or active evasion if an
        #    immune cell is within sensing_radius.
        # =============================================================
        if len(cancer_idx) > 0:
            can_pos = self.positions[cancer_idx]
            if len(immune_idx) > 0:
                imm_pos = self.positions[immune_idx]
                diff = can_pos[:, None, :] - imm_pos[None, :, :]   # (Nc,Ni,2) points AWAY from immune
                dist = np.linalg.norm(diff, axis=2)
            for k, j in enumerate(cancer_idx):
                if self.phenotype[j] == 1 and len(immune_idx) > 0:  # evasive
                    within = dist[k] <= self.sensing_radius[j]
                    if within.any():
                        nearest_local = np.argmin(np.where(within, dist[k], np.inf))
                        flee_dir = diff[k, nearest_local]
                        d = np.linalg.norm(flee_dir) + 1e-6
                        desired_vel = (flee_dir / d) * self.max_speed[j]
                        accel[j] = (desired_vel - self.velocities[j]) / self.tau[j]
                        continue
                # default: sluggish random drift (sessile, or evasive-but-unthreatened)
                accel[j] = rng.normal(0, 0.08, size=2) - 0.1 * self.velocities[j]

        # =============================================================
        # 3) Environmental noise (phenotype-dependent magnitude)
        # =============================================================
        active_idx = np.where(self.active_mask)[0]
        accel[active_idx] += rng.normal(0, 1, size=(len(active_idx), 2)) * self.noise_scale[active_idx, None]

        # =============================================================
        # 4) Soft cell-cell collision repulsion (vectorized pairwise)
        # =============================================================
        if len(active_idx) > 1:
            pos_a = self.positions[active_idx]
            diff = pos_a[:, None, :] - pos_a[None, :, :]        # (n,n,2)
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            close = dist < self.MIN_SEPARATION
            if close.any():
                overlap = np.clip(self.MIN_SEPARATION - dist, 0, None)
                # unit vectors pushing cells apart, weighted by overlap severity
                unit = diff / (dist[:, :, None] + 1e-6)
                repulse = (unit * overlap[:, :, None] * close[:, :, None]).sum(axis=1)
                accel[active_idx] += repulse * 0.5

        # =============================================================
        # 5) Soft boundary repulsion (push away smoothly, no hard bounce)
        # =============================================================
        for axis, extent in enumerate([self.width, self.height]):
            near_low = self.positions[active_idx, axis] < self.BOUNDARY_MARGIN
            near_high = self.positions[active_idx, axis] > extent - self.BOUNDARY_MARGIN
            push = np.zeros(len(active_idx))
            push[near_low] = (self.BOUNDARY_MARGIN - self.positions[active_idx[near_low], axis]) * 0.3
            push[near_high] = -(self.positions[active_idx[near_high], axis] - (extent - self.BOUNDARY_MARGIN)) * 0.3
            accel[active_idx, axis] += push

        # =============================================================
        # 6) Integrate motion: acceleration -> velocity -> position
        #    (real inertia, not instantaneous velocity overwrite)
        # =============================================================
        self.velocities[active_idx] += accel[active_idx] * self.dt
        speed = np.linalg.norm(self.velocities[active_idx], axis=1) + 1e-9
        cap = self.max_speed[active_idx]
        scale = np.minimum(1.0, cap / speed)
        self.velocities[active_idx] *= scale[:, None]
        self.positions[active_idx] += self.velocities[active_idx] * self.dt
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.width)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.height)

        # dead cells: frozen (zero velocity, no movement) -- matches v1 behavior
        dead_idx = np.where(~self.active_mask)[0]
        self.velocities[dead_idx] = 0.0

        # =============================================================
        # 7) Contact tracking + probabilistic, duration-gated killing
        # =============================================================
        immune_idx = np.where((self.base_class == 1) & self.active_mask)[0]
        cancer_idx = np.where((self.base_class == 2) & self.active_mask)[0]
        if len(immune_idx) > 0 and len(cancer_idx) > 0:
            imm_pos = self.positions[immune_idx]
            can_pos = self.positions[cancer_idx]
            diff = can_pos[:, None, :] - imm_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=2)                 # (Nc, Ni)
            nearest_imm_local = np.argmin(dist, axis=1)
            nearest_dist = dist[np.arange(len(cancer_idx)), nearest_imm_local]
            in_contact = nearest_dist < self.KILL_RADIUS

            for k, j in enumerate(cancer_idx):
                if in_contact[k]:
                    self.contact_timer[j] += 1
                else:
                    self.contact_timer[j] = 0

                if self.contact_timer[j] >= self.ENGAGE_STEPS_REQUIRED:
                    attacker = immune_idx[nearest_imm_local[k]]
                    p_kill = self.kill_rate[attacker] * (0.5 + 0.5 * self.energy[attacker])
                    if rng.random() < p_kill:
                        self.active_mask[j] = False
                        self.contact_timer[j] = 0
                        self.locked_target[self.locked_target == j] = -1
                        self.energy[attacker] = min(1.0, self.energy[attacker] + 0.15)  # brief rebound

        # =============================================================
        # 8) Write this frame (same 5-feature layout as v1)
        # =============================================================
        self.trajectory_data[:, t, 0:2] = self.positions
        self.trajectory_data[:, t, 2:4] = self.velocities
        self.trajectory_data[:, t, 4] = self._type_column()

    def run_simulation(self):
        for t in range(self.timesteps):
            self.step(t)
        return self.trajectory_data


# ==========================================
# DATASET GENERATION (unchanged interface)
# ==========================================
def generate_dataset(mode, run_id):
    """Runs one simulation and saves it exactly as v1 did, so the existing
    hopfield_timeframe.ipynb (data_dir='simulation_data', filenames
    data_{mode}_{i}.npy) works with zero changes."""
    output_dir = "new_simulation_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # seed so each (mode, run_id) is independently reproducible
    seed = hash((mode, run_id)) % (2**32)
    sim = CellSimulation(num_immune=8, num_cancer=12, timesteps=600, mode=mode, seed=seed)
    trajectories = sim.run_simulation()

    filename = os.path.join(output_dir, f"data_{mode}_{run_id}.npy")
    np.save(filename, trajectories)


if __name__ == "__main__":
    print("Generating 50 Killing and 50 Non-Killing simulations inside 'simulation_data/'...")
    for i in range(50):
        generate_dataset('killing', i)
        generate_dataset('non-killing', i)
    print("All 100 simulation files successfully generated!")