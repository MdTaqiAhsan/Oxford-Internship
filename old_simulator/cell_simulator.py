import numpy as np

class CellSimulation:
    def __init__(self, num_immune=10, num_cancer=15, width=100, height=100, timesteps=100, mode='killing'):
        """
        mode: 'killing' (Type A) or 'non-killing' (Type B)
        cell_type: 1 for Immune, 2 for Cancer
        """
        self.num_immune = num_immune
        self.num_cancer = num_cancer
        self.num_cells = num_immune + num_cancer
        self.width = width
        self.height = height
        self.timesteps = timesteps
        self.mode = mode
        
        # Initialize positions randomly
        self.positions = np.random.uniform(0, [width, height], size=(self.num_cells, 2))
        
        # Initialize velocities randomly
        self.velocities = np.random.uniform(-1, 1, size=(self.num_cells, 2))
        
        # Cell types: 1 = Immune, 2 = Cancer
        self.types = np.array([1] * num_immune + [2] * num_cancer)
        
        # Mask to track active/alive cells (1 = Alive, 0 = Dead/Inactive)
        self.active_mask = np.ones(self.num_cells)
        
        # Master trajectory array: [N_cells, timesteps, features]
        # Features: [x, y, vx, vy, type]
        self.trajectory_data = np.zeros((self.num_cells, timesteps, 5))

    def step(self, t):
        """Advances the simulation by one timestep."""
        for i in range(self.num_cells):
            if not self.active_mask[i]:
                # If a cell is dead/inactive, it stays frozen at its last position with 0 velocity
                self.velocities[i] = [0, 0]
                continue
                
            if self.types[i] == 1:  # Immune Cell Logic
                if self.mode == 'killing':
                    # Find the closest active cancer cell
                    cancer_indices = np.where((self.types == 2) & (self.active_mask == 1))[0]
                    if len(cancer_indices) > 0:
                        distances = np.linalg.norm(self.positions[cancer_indices] - self.positions[i], axis=1)
                        closest_cancer = cancer_indices[np.argmin(distances)]
                        
                        # Direct vector towards the closest cancer cell (tracking behavior)
                        direction = self.positions[closest_cancer] - self.positions[i]
                        direction /= (np.linalg.norm(direction) + 1e-5)  # normalize
                        self.velocities[i] = direction * 1.5  # Immune cells move slightly faster
                else:
                    # Non-killing mode: Random brownian-like movement
                    self.velocities[i] += np.random.uniform(-0.2, 0.2, size=2)
                    self.velocities[i] = np.clip(self.velocities[i], -1, 1)
                    
            elif self.types[i] == 2:  # Cancer Cell Logic
                # Cancer cells just drift randomly/sluggishly
                self.velocities[i] += np.random.uniform(-0.1, 0.1, size=2)
                self.velocities[i] = np.clip(self.velocities[i], -0.5, 0.5)

            # Update position
            self.positions[i] += self.velocities[i]
            
            # Boundary conditions (bounce off the walls)
            for axis in range(2):
                if self.positions[i, axis] <= 0 or self.positions[i, axis] >= [self.width, self.height][axis]:
                    self.velocities[i, axis] *= -1
                    self.positions[i, axis] = np.clip(self.positions[i, axis], 0, [self.width, self.height][axis])

        # Handle interaction/collision checks (Only relevant in killing mode)
        if self.mode == 'killing':
            immune_indices = np.where(self.types == 1)[0]
            cancer_indices = np.where((self.types == 2) & (self.active_mask == 1))[0]
            
            for imm in immune_indices:
                for canc in cancer_indices:
                    dist = np.linalg.norm(self.positions[imm] - self.positions[canc])
                    if dist < 2.5:  # Collision radius for interaction
                        self.active_mask[canc] = 0  # Cancer cell is killed

        # Save state to trajectory master array
        self.trajectory_data[:, t, 0:2] = self.positions
        self.trajectory_data[:, t, 2:4] = self.velocities
        self.trajectory_data[:, t, 4] = self.types * self.active_mask # Type drops to 0 if dead

    def run_simulation(self):
        """Runs the loop for all timesteps."""
        for t in range(self.timesteps):
            self.step(t)
        return self.trajectory_data

# ==========================================
# EXECUTION & ANIMATION VISUALIZATION
# ==========================================
import os

def generate_dataset(mode, run_id):
    """Helper function to run a single simulation and save it with a unique ID inside a folder."""
    # Create the folder if it doesn't exist yet
    output_dir = "simulation_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sim = CellSimulation(num_immune=8, num_cancer=12, timesteps=120, mode=mode)
    trajectories = sim.run_simulation()
    
    # Save with unique identifier inside the folder
    filename = os.path.join(output_dir, f"data_{mode}_{run_id}.npy")
    np.save(filename, trajectories)

if __name__ == "__main__":
    print("Generating 50 Killing and 50 Non-Killing simulations inside 'simulation_data/'...")
    for i in range(50):
        generate_dataset('killing', i)
        generate_dataset('non-killing', i)
    print("All 100 simulation files successfully generated!")

# What I'd improve

# A few changes would make the simulation more biologically realistic and more useful for machine learning:

# Add persistence to movement instead of abruptly changing velocity.
# Allow cancer cells to actively evade nearby immune cells.
# Introduce randomness into immune-cell pursuit rather than perfect tracking.
# Include interaction delay (immune cells should need to remain in contact for some time before killing).
# Replace hard kill/no-kill with probabilistic killing.
# Add acceleration and cell mass rather than directly overwriting velocity.
# Introduce obstacles or tissue boundaries instead of an empty box.
# Record additional features such as distance to nearest target, interaction duration, or local cell density, which could improve downstream learning.
