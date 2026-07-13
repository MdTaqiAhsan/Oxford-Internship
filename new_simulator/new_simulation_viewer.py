import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Type encoding produced by cell_simulator.py (v2):
#   0.0 -> dead / inactive
#   1.0 -> immune, phenotype "scout"      (fast, weak kill)
#   2.0 -> immune, phenotype "cytotoxic"  (slow, strong kill)
#   3.0 -> cancer, phenotype "sessile"    (low motility)
#   4.0 -> cancer, phenotype "evasive"    (flees immune cells)


def view_simulation(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    trajectories = np.load(file_path)
    num_cells, timesteps, feature_dim = trajectories.shape

    print(f"Successfully loaded: {file_path}")
    print(f"Cells: {num_cells} | Timesteps: {timesteps}")

    file_name = os.path.basename(file_path)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"Visualizing Run: {file_name}", fontsize=12, fontweight='bold')

    # One scatter artist per phenotype so each gets its own color/marker,
    # plus a faded one for dead cells so you can see where kills happened.
    scout_scatter = ax.scatter([], [], c='deepskyblue', label='Immune: scout', s=45,
                                edgecolors='black', alpha=0.85)
    cytotoxic_scatter = ax.scatter([], [], c='deepskyblue', label='Immune: cytotoxic', s=55,
                                    edgecolors='black', alpha=0.85)
    sessile_scatter = ax.scatter([], [], c='pink', label='Cancer: sessile', s=65,
                                  edgecolors='black', alpha=0.85)
    evasive_scatter = ax.scatter([], [], c='pink', label='Cancer: evasive', s=65,
                                  edgecolors='black', alpha=0.85)
    dead_scatter = ax.scatter([], [], c='yellow', label='Dead', s=40,
                               edgecolors='black', alpha=0.5)

    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    timestep_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             va='top', ha='left', fontsize=9,
                             bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    def update(frame):
        current_frame_data = trajectories[:, frame, :]
        type_col = current_frame_data[:, 4]

        scout_scatter.set_offsets(current_frame_data[type_col == 1, 0:2])
        cytotoxic_scatter.set_offsets(current_frame_data[type_col == 2, 0:2])
        sessile_scatter.set_offsets(current_frame_data[type_col == 3, 0:2])
        evasive_scatter.set_offsets(current_frame_data[type_col == 4, 0:2])
        dead_scatter.set_offsets(current_frame_data[type_col == 0, 0:2])

        n_cancer_alive = int(np.isin(type_col, [3, 4]).sum())
        timestep_text.set_text(f"t = {frame}/{timesteps - 1}\ncancer alive: {n_cancer_alive}")

        return (scout_scatter, cytotoxic_scatter, sessile_scatter,
                evasive_scatter, dead_scatter, timestep_text)

    ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=50,
                                   blit=True, repeat=True)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage error! Please provide a file trajectory path.")
        print(r"Example: python new_simulator\new_simulation_viewer.py new_simulator\new_simulation_data\data_killing_0.npy")
    else:
        target_file = sys.argv[1]
        view_simulation(target_file)
#python new_simulator\new_simulation_viewer.py new_simulator\new_simulation_data\data_killing_0.npy        