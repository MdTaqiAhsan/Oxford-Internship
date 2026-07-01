import torch
import torch.optim as optim
import numpy as np
import os
from hflayers import Hopfield

# 1. LOAD AND COMPUTE THE TRAINING MEMORY BANK (Same as before)
print("Compiling unified memory bank from training data (0-34)...")
train_memories = []
data_dir = "simulation_data"

for i in range(35):
    kill_data = np.load(os.path.join(data_dir, f"data_killing_{i}.npy"))
    train_memories.append(torch.tensor(kill_data, dtype=torch.float32).permute(1, 0, 2).reshape(120, 100))
    nk_data = np.load(os.path.join(data_dir, f"data_non-killing_{i}.npy"))
    train_memories.append(torch.tensor(nk_data, dtype=torch.float32).permute(1, 0, 2).reshape(120, 100))

memory_bank = torch.cat(train_memories, dim=0)
K = memory_bank.unsqueeze(0)
V = memory_bank.unsqueeze(0)

# Initialize the Network
chnn_layer = Hopfield(
    input_size=100, stored_pattern_size=100, pattern_projection_size=100, scaling=10.0,
    state_pattern_as_static=True, stored_pattern_as_static=True, pattern_projection_as_static=True,
    normalize_stored_pattern=False, normalize_stored_pattern_affine=False,
    normalize_state_pattern=False, normalize_state_pattern_affine=False,
    normalize_pattern_projection=False, normalize_pattern_projection_affine=False,
    disable_out_projection=True
)

# 2. EVALUATION LOOP FOR ALL TEST FILES (35 to 49)
test_modes = ['killing', 'non-killing']
total_cells_evaluated = 0
correct_fate_predictions = 0

print("\nStarting comprehensive evaluation on test sets (runs 35-49)...")

for mode in test_modes:
    mode_correct = 0
    mode_total = 0
    
    for run_id in range(35, 50):
        # Load test file
        test_path = os.path.join(data_dir, f"data_custom_{mode}_{run_id}.npy") if os.path.exists(os.path.join(data_dir, f"data_custom_{mode}_{run_id}.npy")) else os.path.join(data_dir, f"data_{mode}_{run_id}.npy")
        test_raw = np.load(test_path)
        clean_ground_truth = torch.tensor(test_raw, dtype=torch.float32).permute(1, 0, 2).reshape(120, 100)
        
        # Setup mask parameters
        mask_start = 80
        corrupted_query = clean_ground_truth.clone()
        trainable_mask = torch.randn_like(corrupted_query[mask_start:, :]) * 5.0
        trainable_mask.requires_grad_(True)
        
        optimizer = optim.Adam([trainable_mask], lr=0.2)
        
        # Short optimized backpropagation pass per file
        for epoch in range(61):
            optimizer.zero_grad()
            current_query = torch.cat([corrupted_query[:mask_start, :], trainable_mask], dim=0)
            reconstructed_state = chnn_layer((K, current_query.unsqueeze(0), V)).squeeze(0)
            loss = torch.mean((current_query - reconstructed_state) ** 2)
            loss.backward()
            optimizer.step()
            
        # Evaluate final time step fates for this file
        final_query = torch.cat([corrupted_query[:mask_start, :], trainable_mask], dim=0)
        predicted_status = final_query[-1, 4::5].detach().numpy()
        actual_status = clean_ground_truth[-1, 4::5].numpy()
        
        # Binary classification tracking (0 = Killed, 1 or 2 = Alive)
        actual_fates = np.where(actual_status == 0, 0, 1)
        predicted_fates = np.where(predicted_status < 0.5, 0, 1)
        
        matches = np.sum(actual_fates == predicted_fates)
        mode_correct += matches
        mode_total += len(actual_fates)
        
    print(f" > Mode [{mode.upper()}] Accuracy: {mode_correct}/{mode_total} ({mode_correct/mode_total*100:.1f}%)")
    correct_fate_predictions += mode_correct
    total_cells_evaluated += mode_total

# 3. FINAL SUMMARY
aggregate_accuracy = (correct_fate_predictions / total_cells_evaluated) * 100
print(f"\n================ FINAL AGGREGATE PERFORMANCE ================")
print(f"Total Cells Evaluated: {total_cells_evaluated}")
print(f"Overall Fate Classification Accuracy: {aggregate_accuracy:.2f}%")