import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def view_simulation(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Load the trajectory array: (num_cells, timesteps, features)
    trajectories = np.load(file_path)
    num_cells, timesteps, feature_dim = trajectories.shape
    
    print(f"Successfully loaded: {file_path}")
    print(f"Cells: {num_cells} | Timesteps: {timesteps}")
    
    # Extract simulation mode/identity from filename for the title
    file_name = os.path.basename(file_path)
    
    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"Visualizing Run: {file_name}", fontsize=12, fontweight='bold')
    
    # Create distinct scatter plot elements for active types
    immune_scatter = ax.scatter([], [], c='blue', label='Immune Cells (T-Cells)', s=50, edgecolors='black', alpha=0.8)
    cancer_scatter = ax.scatter([], [], c='red', label='Cancer Cells', s=70, edgecolors='black', alpha=0.8)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)

    def update(frame):
        # Extract features for all cells at the current timestep frame
        current_frame_data = trajectories[:, frame, :]
        
        # Feature index 4 holds the cell state mask (1=Immune, 2=Cancer, 0=Killed/Inactive)
        imm_idx = np.where(current_frame_data[:, 4] == 1)[0]
        canc_idx = np.where(current_frame_data[:, 4] == 2)[0]
        
        # Update plot coordinates dynamically
        immune_scatter.set_offsets(current_frame_data[imm_idx, 0:2])
        cancer_scatter.set_offsets(current_frame_data[canc_idx, 0:2])
        return immune_scatter, cancer_scatter

    # Build the animation engine loop
    ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=50, blit=True, repeat=True)
    plt.show()

if __name__ == "__main__":
    # Check if a filename was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage error! Please provide a file trajectory path.")
        print(r"Example: python simulation_viewer.py simulation_data\data_killing_0.npy")
    else:
        target_file = sys.argv[1]
        view_simulation(target_file)