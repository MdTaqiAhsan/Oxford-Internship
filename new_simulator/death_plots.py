import numpy as np
import matplotlib.pyplot as plt

# File paths for both datasets
killing_path = "new_simulator/new_simulation_data/data_killing_0.npy"
non_killing_path = "new_simulator/new_simulation_data/data_non-killing_0.npy"

def extract_population_stats(filepath):
    data = np.load(filepath)  # Shape: (num_cells, timesteps, 5)
    timesteps = data.shape[1]
    
    immune_alive = []
    cancer_alive = []
    total_dead = []
    
    for t in range(timesteps):
        types = data[:, t, 4]
        is_dead = types == 0.0
        is_immune = (types >= 1.0) & (types < 3.0)
        is_cancer = types >= 3.0
        
        total_dead.append(np.sum(is_dead))
        immune_alive.append(np.sum(is_immune))
        cancer_alive.append(np.sum(is_cancer))
        
    return np.array(immune_alive), np.array(cancer_alive), np.array(total_dead)

# Extract metrics
k_immune, k_cancer, k_dead = extract_population_stats(killing_path)
nk_immune, nk_cancer, nk_dead = extract_population_stats(non_killing_path)

timesteps = len(k_immune)
time_axis = np.arange(timesteps)

# Setup comparison plot styling
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
fig.patch.set_facecolor('#0E1117')

for ax in axes:
    ax.set_facecolor("#FAFAFA")
    ax.tick_params(colors='white', labelsize=10)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, color='#2A2E39', linestyle='--', linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')

# Plot 1: Killing Mode
axes[0].plot(time_axis, k_immune, label="Immune Cells (Alive)", color="#87CEFA", linewidth=2.2)
axes[0].plot(time_axis, k_cancer, label="Cancer Cells (Alive)", color="#FF3131", linewidth=2.2)
axes[0].plot(time_axis, k_dead, label="Total Dead Cells", color="#8A8A8A", linestyle="--", linewidth=1.5)
axes[0].set_title("Killing Mode Dynamics", fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel("Time (timesteps)")
axes[0].set_ylabel("Cell Count")
axes[0].legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white", fontsize=9)

# Plot 2: Non-Killing Mode
axes[1].plot(time_axis, nk_immune, label="Immune Cells (Alive)", color="#87CEFA", linewidth=2.2)
axes[1].plot(time_axis, nk_cancer, label="Cancer Cells (Alive)", color="#FF3131", linewidth=2.2)
axes[1].plot(time_axis, nk_dead, label="Total Dead Cells", color="#8A8A8A", linestyle="--", linewidth=1.5)
axes[1].set_title("Non-Killing Mode Dynamics", fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel("Time (timesteps)")
axes[1].set_ylabel("Cell Count")
axes[1].legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white", fontsize=9)

plt.tight_layout()
plt.show()

# Print text summary to console
print("==========================================")
print("         SIMULATION STATS SUMMARY         ")
print("==========================================")
print(f"[Killing Mode]")
print(f" • Final Cancer Alive : {k_cancer[-1]}")
print(f" • Final Immune Alive : {k_immune[-1]}")
print(f" • Cumulative Casualties: {k_dead[-1]}")
print("------------------------------------------")
print(f"[Non-Killing Mode]")
print(f" • Final Cancer Alive : {nk_cancer[-1]}")
print(f" • Final Immune Alive : {nk_immune[-1]}")
print(f" • Cumulative Casualties: {nk_dead[-1]}")
print("==========================================")