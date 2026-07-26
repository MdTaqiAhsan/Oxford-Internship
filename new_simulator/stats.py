import numpy as np
import matplotlib.pyplot as plt

killing_path = "new_simulator/new_simulation_data/data_killing_0.npy"
non_killing_path = "new_simulator/new_simulation_data/data_non-killing_0.npy"

# Subtype Encodings
PHENOTYPES = {
    1.0: "Scout",
    1.4: "Messenger",
    1.8: "Killer",
    3.0: "Sessile Cancer",
    4.0: "Evasive Cancer"
}

def analyze_dataset(filepath):
    data = np.load(filepath)  # Shape: (N, T, 5) -> [x, y, vx, vy, type]
    N, T, _ = data.shape
    
    # 1. Velocities & Speeds
    vx = data[:, :, 2]
    vy = data[:, :, 3]
    speeds = np.sqrt(vx**2 + vy**2)  # Shape: (N, T)
    types = data[:, :, 4]
    
    # Average speed per subtype
    subtype_speeds = {}
    for code, label in PHENOTYPES.items():
        mask = np.isclose(types, code, atol=0.1)
        if np.any(mask):
            subtype_speeds[label] = np.mean(speeds[mask])
        else:
            subtype_speeds[label] = 0.0

    # 2. Total Path Length & Net Displacement
    positions = data[:, :, 0:2]  # (N, T, 2)
    step_dists = np.linalg.norm(np.diff(positions, axis=1), axis=2)  # (N, T-1)
    total_path_length = np.sum(step_dists, axis=1)  # (N,)
    
    net_displacement = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=1)
    
    # Tortuosity / Straightness index (1.0 = direct straight line, 0.0 = random walk)
    straightness = np.where(total_path_length > 0, net_displacement / total_path_length, 0.0)

    # 3. Population Dynamics & Kill Rates
    cancer_mask = types >= 3.0
    cancer_counts = np.sum(cancer_mask, axis=0)  # Count per frame
    
    initial_cancer = cancer_counts[0]
    final_cancer = cancer_counts[-1]
    total_kills = initial_cancer - final_cancer
    overall_kill_rate = (total_kills / initial_cancer) * 100 if initial_cancer > 0 else 0.0
    
    # Instantaneous Kill Rate (Kills per minute)
    instant_kill_rate = -np.diff(cancer_counts)  # Positive when cancer dies
    
    # 4. Spatial Clustering (Mean Nearest Neighbor Distance at final frame)
    final_pos = positions[:, -1, :]
    final_active = types[:, -1] > 0
    active_pos = final_pos[final_active]
    
    if len(active_pos) > 1:
        # Pairwise distance matrix
        dist_matrix = np.linalg.norm(active_pos[:, None, :] - active_pos[None, :, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)
        mean_nnd = np.mean(np.min(dist_matrix, axis=1))
    else:
        mean_nnd = 0.0

    return {
        "cancer_counts": cancer_counts,
        "instant_kill_rate": instant_kill_rate,
        "overall_kill_rate": overall_kill_rate,
        "total_kills": total_kills,
        "subtype_speeds": subtype_speeds,
        "mean_speed_overall": np.mean(speeds[types > 0]),
        "mean_straightness": np.mean(straightness[types[:, 0] > 0]),
        "mean_nnd": mean_nnd,
        "mean_path_length": np.mean(total_path_length[types[:, 0] > 0])
    }

# Execute metrics computation
stats_k = analyze_dataset(killing_path)
stats_nk = analyze_dataset(non_killing_path)

# ==========================================
# PRINT DETAILED COMPARATIVE SUMMARY
# ==========================================
print("=====================================================================")
print("                  ADVANCED SIMULATION METRICS                        ")
print("=====================================================================")
print(f"{'Metric':<35} | {'Killing Mode':<15} | {'Non-Killing Mode':<15}")
print("---------------------------------------------------------------------")
print(f"{'Total Cancer Eliminated':<35} | {stats_k['total_kills']:<15} | {stats_nk['total_kills']:<15}")
print(f"{'Overall Clearance (%)':<35} | {stats_k['overall_kill_rate']:<14.2f}% | {stats_nk['overall_kill_rate']:<14.2f}%")
print(f"{'Peak Instant Kill Rate (cells/timestep)':<35} | {np.max(stats_k['instant_kill_rate']):<15} | {np.max(stats_nk['instant_kill_rate']):<15}")
print(f"{'Mean Cell Speed (μm/timestep)':<35} | {stats_k['mean_speed_overall']:<15.2f} | {stats_nk['mean_speed_overall']:<15.2f}")
print(f"{'Mean Path Length (μm)':<35} | {stats_k['mean_path_length']:<15.2f} | {stats_nk['mean_path_length']:<15.2f}")
print(f"{'Trajectory Straightness (0-1)':<35} | {stats_k['mean_straightness']:<15.3f} | {stats_nk['mean_straightness']:<15.3f}")
print(f"{'Final Nearest-Neighbor Dist (μm)':<35} | {stats_k['mean_nnd']:<15.2f} | {stats_nk['mean_nnd']:<15.2f}")
print("---------------------------------------------------------------------")
print("\n--- MEAN SPEED BY PHENOTYPE SUBTYPE (μm/timestep) ---")
for label in PHENOTYPES.values():
    sp_k = stats_k['subtype_speeds'].get(label, 0.0)
    sp_nk = stats_nk['subtype_speeds'].get(label, 0.0)
    print(f" • {label:<22} : Killing = {sp_k:.2f} | Non-Killing = {sp_nk:.2f}")
print("=====================================================================")

# ==========================================
# PLOT STATISTICAL DIAGNOSTIC DASHBOARD
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
fig.patch.set_facecolor('#0E1117')

for ax in axes:
    ax.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, color='#2A2E39', linestyle='--', linewidth=0.5)

# Plot 1: Instantaneous Kill Rate Over Time
time_axis = np.arange(len(stats_k['instant_kill_rate']))
axes[0].plot(time_axis, stats_k['instant_kill_rate'], color='#39FF14', label='Killing Mode', linewidth=1.8)
axes[0].plot(time_axis, stats_nk['instant_kill_rate'], color='#FF3131', linestyle='--', label='Non-Killing Mode', linewidth=1.5)
axes[0].set_title("Instantaneous Clearance Rate (-dCancer/dt)", fontweight='bold')
axes[0].set_xlabel("Time (minutes)")
axes[0].set_ylabel("Cancer Kills / Frame")
axes[0].legend(facecolor="#161B22", labelcolor="white")

# Plot 2: Subtype Speed Distribution Comparison
labels = list(PHENOTYPES.values())
x = np.arange(len(labels))
width = 0.35

k_speeds = [stats_k['subtype_speeds'][l] for l in labels]
nk_speeds = [stats_nk['subtype_speeds'][l] for l in labels]

axes[1].bar(x - width/2, k_speeds, width, label='Killing', color='#00BAFF')
axes[1].bar(x + width/2, nk_speeds, width, label='Non-Killing', color='#8A2BE2')
axes[1].set_title("Average Movement Speed per Subtype", fontweight='bold')
axes[1].set_ylabel("Speed (μm/timestep)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=20, ha='right', fontsize=9, color='white')
axes[1].legend(facecolor="#161B22", labelcolor="white")

plt.tight_layout()
plt.show()