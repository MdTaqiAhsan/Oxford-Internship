import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# 1. Load simulation dataset
data_path = "new_simulator/new_simulation_data/data_killing_0.npy"
trajectories = np.load(data_path)  # Shape: (num_cells, timesteps, 5)

num_cells, timesteps, _ = trajectories.shape

# 2. Phenotype dictionary with your custom color mapping
PHENOTYPE_INFO = {
    0.0: {"color": "#4A4A4A", "label": "Dead Cell", "size": 6},
    1.0: {"color": "#87CEFA", "label": "Immune Scout", "size": 18},      # Light Blue
    1.4: {"color": "#008080", "label": "Immune Messenger", "size": 18},  # Teal
    1.8: {"color": "#00008B", "label": "Immune Killer", "size": 22},     # Dark Blue
    3.0: {"color": "#FF8C00", "label": "Normal (Sessile) Cancer", "size": 20}, # Orange
    4.0: {"color": "#FF0000", "label": "Evasive Cancer", "size": 20},    # Red
}

fig, ax = plt.subplots(figsize=(9, 8), dpi=120)
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#EEEEEE")

# Determine domain boundaries
max_x = np.max(trajectories[:, :, 0])
max_y = np.max(trajectories[:, :, 1])
ax.set_xlim(0, max_x)
ax.set_ylim(0, max_y)

# Axis Labelling & Styling
ax.set_title("Immune-Cancer Microenvironment Simulation", fontsize=14, fontweight='bold', color='white', pad=15)
ax.set_xlabel("Spatial Coordinate X (μm)", fontsize=11, color='white')
ax.set_ylabel("Spatial Coordinate Y (μm)", fontsize=11, color='white')
ax.tick_params(colors='white', labelsize=9)
ax.grid(True, color='#2A2E39', linestyle='--', linewidth=0.5, alpha=0.7)

# 3. Create Color-Coded Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=info["label"],
           markerfacecolor=info["color"], markersize=np.sqrt(info["size"]), linestyle='None')
    for info in PHENOTYPE_INFO.values()
]
legend = ax.legend(handles=legend_elements, loc="upper right", facecolor="#161B22", 
                   edgecolor="#30363D", labelcolor="white", fontsize=9, framealpha=0.9)

# 4. On-Screen Live Stat Overlay
stat_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white", 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#161B22", edgecolor="#30363D", alpha=0.8))

# Initialize scatter plot
scatter = ax.scatter([], [], alpha=0.9)

def get_visual_properties(types):
    colors = []
    sizes = []
    for t in types:
        # Snap floating encoding to nearest phenotype key
        key = min(PHENOTYPE_INFO.keys(), key=lambda k: abs(k - t))
        colors.append(PHENOTYPE_INFO[key]["color"])
        sizes.append(PHENOTYPE_INFO[key]["size"])
    return colors, sizes

def update(frame):
    pos = trajectories[:, frame, 0:2]
    types = trajectories[:, frame, 4]
    
    colors, sizes = get_visual_properties(types)
    scatter.set_offsets(pos)
    scatter.set_color(colors)
    scatter.set_sizes(sizes)
    
    # Calculate live counts
    immune_alive = np.sum((types >= 1.0) & (types < 3.0))
    cancer_alive = np.sum(types >= 3.0)
    total_dead = np.sum(types == 0.0)
    
    # Update live label text
    stat_text.set_text(
        f"Time: {frame}\n"
        f"• Immune Alive: {immune_alive}\n"
        f"• Cancer Alive: {cancer_alive}\n"
        f"• Total Deaths: {total_dead}"
    )
    
    return scatter, stat_text

anim = animation.FuncAnimation(fig, update, frames=timesteps, interval=60, blit=True)

# To save as video:
# anim.save("labeled_simulation.mp4", writer="ffmpeg", fps=20)

plt.tight_layout()
plt.show()