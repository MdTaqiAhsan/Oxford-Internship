import numpy as np
import matplotlib.pyplot as plt

# File paths for both datasets
killing_path = "new_simulator/new_simulation_data/data_killing_0.npy"
non_killing_path = "new_simulator/new_simulation_data/data_non-killing_0.npy"

def extract_death_stats(filepath):
    data = np.load(filepath)  # Shape: (num_cells, timesteps, 5)
    timesteps = data.shape[1]

    # Initial population counts at t=0
    init_types = data[:, 0, 4]
    init_immune = np.sum((init_types >= 1.0) & (init_types < 3.0))
    init_cancer = np.sum(init_types >= 3.0)

    immune_deaths = []
    cancer_deaths = []

    for t in range(timesteps):
        types = data[:, t, 4]
        alive_immune = np.sum((types >= 1.0) & (types < 3.0))
        alive_cancer = np.sum(types >= 3.0)

        # Cumulative deaths = initial count - current alive count
        immune_deaths.append(init_immune - alive_immune)
        cancer_deaths.append(init_cancer - alive_cancer)

    return np.array(immune_deaths), np.array(cancer_deaths)

# Extract cumulative deaths
k_immune_deaths, k_cancer_deaths = extract_death_stats(killing_path)
nk_immune_deaths, nk_cancer_deaths = extract_death_stats(non_killing_path)

timesteps = len(k_immune_deaths)
time_axis = np.arange(timesteps)

# Setup comparison plot styling
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
fig.patch.set_facecolor('#0E1117')

for ax in axes:
    ax.set_facecolor("#161B22")
    ax.tick_params(colors='white', labelsize=10)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, color='#2A2E39', linestyle='--', linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')

# Panel 1: Immune Cell Deaths (Killing vs Non-Killing)
axes[0].plot(time_axis, k_immune_deaths, label="Killing Mode", color="#FF3131", linewidth=2.2)
axes[0].plot(time_axis, nk_immune_deaths, label="Non-Killing Mode", color="#2ca02c", linewidth=2.2)
axes[0].set_title("Immune Cell Cumulative Deaths", fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel("Time (timesteps)")
axes[0].set_ylabel("Cumulative Deaths")
axes[0].legend(facecolor="#0E1117", edgecolor="#30363D", labelcolor="white", fontsize=9.5)

# Panel 2: Cancer Cell Deaths (Killing vs Non-Killing)
axes[1].plot(time_axis, k_cancer_deaths, label="Killing Mode", color="#FF3131", linewidth=2.2)
axes[1].plot(time_axis, nk_cancer_deaths, label="Non-Killing Mode", color="#2ca02c", linewidth=2.2)
axes[1].set_title("Cancer Cell Cumulative Deaths", fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel("Time (timesteps)")
axes[1].set_ylabel("Cumulative Deaths")
axes[1].legend(facecolor="#0E1117", edgecolor="#30363D", labelcolor="white", fontsize=9.5)

plt.tight_layout()
plt.show()

# Print text summary to console
print("==========================================")
print("       CUMULATIVE DEATH STATS SUMMARY     ")
print("==========================================")
print(f"[Immune Cell Deaths]")
print(f" • Killing Mode     : {k_immune_deaths[-1]}")
print(f" • Non-Killing Mode : {nk_immune_deaths[-1]}")
print("------------------------------------------")
print(f"[Cancer Cell Deaths]")
print(f" • Killing Mode     : {k_cancer_deaths[-1]}")
print(f" • Non-Killing Mode : {nk_cancer_deaths[-1]}")
print("==========================================")