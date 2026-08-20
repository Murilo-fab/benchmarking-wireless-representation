"""
Beam Prediction / Selection Task.

This script benchmarks the model's ability to select the optimal beam index 
from a codebook based on the channel estimate. It evaluates:
1. Data Efficiency: F1 Score vs. Training Size.
2. Noise Robustness: F1 Score vs. SNR (using fixed noise floor physics).
3. Computational Complexity.

Author: Murilo Ferreira Alves Batista - RWTH Aachen/USP
"""

# --- 1. Standard Library Imports ---
import os
import csv
import time
import copy
from datetime import datetime

# --- 2. Third-Party Imports ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from tqdm import tqdm
import DeepMIMOv3

# --- 3. Local Imports ---
from .data_classes import TaskConfig
from .downstream_models import build_model_from_config
from .utils import (
    get_parameters, 
    create_dataloaders, 
    apply_awgn, 
    extract_features,
    get_flops_and_params, 
    get_latency, 
    get_subset
)

# -----------------------------------------------------------------------------
# CLASS: Beam Prediction Generator
# -----------------------------------------------------------------------------

class BeamPredictionGenerator:
    """
    Handles data generation for Beam Prediction.
    
    Features:
    - Caches raw channel matrices to allow fast re-labeling.
    - Dynamically generates labels based on the requested codebook size (n_beams).
    """
    def __init__(self, scenario_name: str, scale_factor: float = 1e6):
        self.params = get_parameters(scenario_name)
        self.scale_factor = scale_factor
        
        # Cache for the clean data
        self.clean_chs = None 
        self.clean_locs = None

    def load_raw_data(self):
        """
        Loads and filters data from ALL Base Stations. 
        Only keeps users where LoS != -1 and concatenates them into a single dataset.
        """
        print(f"Loading Raw Data for {self.params['scenario']}...")
        
        # 1. Generate Raw Data (returns a list of BS dictionaries)
        deepmimo_data = DeepMIMOv3.generate_data(self.params)
        
        all_bs_chs = []
        all_bs_locs = []
        
        # Iterate through each Base Station in the data list
        for bs_idx, bs_entry in enumerate(deepmimo_data):
            user_data = bs_entry['user']
            
            raw_chs = user_data['channel']
            raw_los = user_data['LoS']
            raw_loc = user_data['location']
            
            # 2. Identify Valid Indices for THIS Base Station
            valid_idxs = np.where(raw_los != -1)[0]
            
            if len(valid_idxs) > 0:
                all_bs_chs.append(raw_chs[valid_idxs])
                all_bs_locs.append(raw_loc[valid_idxs])
            else:
                print(f"\tBS {bs_idx}: No valid users found.")

        # 3. Concatenate and Store
        if not all_bs_chs:
            raise ValueError("No valid users found across any Base Stations.")

        # Stack all collected BS data along the first axis (N_Samples)
        self.clean_chs = np.concatenate(all_bs_chs, axis=0)
        self.clean_locs = np.concatenate(all_bs_locs, axis=0)
        
        print(f"\tTotal Valid User-BS links found: {len(self.clean_chs)}")
        return self.clean_chs

    def compute_labels_for_beams(self, n_beams: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Generates (X, y) pairs for a specific codebook size.
        
        Args:
            n_beams (int): Size of the beam codebook (e.g., 32, 64).
            
        Returns:
            X (Tensor): Complex channels.
            y (Tensor): Beam indices.
            locs (Array): User locations.
        """
        if self.clean_chs is None:
            self.load_raw_data()

        # 1. Compute Labels via Beam Sweeping
        beam_indices, valid_mask = self._compute_beam_labels(self.clean_chs, n_beams)
        
        # 2. Final Filter (removing users with 0 power / NaN beams)
        final_chs = self.clean_chs[valid_mask]
        final_labels = beam_indices[valid_mask]
        final_locs = self.clean_locs[valid_mask]
        
        # 3. Format Dimensions
        if final_chs.ndim == 4:
            final_chs = final_chs.squeeze(axis=1)
            
        X = torch.tensor(final_chs, dtype=torch.complex64) * self.scale_factor
        y = torch.tensor(final_labels, dtype=torch.long)
        
        return X, y, final_locs

    def _compute_beam_labels(self, channels, n_beams):
        """
        Performs exhaustive beam search (DFT Codebook) to find the best beam index.
        """
        n_users = len(channels)
        
        # --- Codebook Setup ---
        fov = 180
        beam_angles = np.around(np.arange(-fov/2, fov/2+.1, fov/(n_beams-1)), 2)
        bs_ant_params = self.params['bs_antenna']
        kd_val = 2 * np.pi * bs_ant_params['spacing']
        
        codebook_list = []
        for azi in beam_angles:
            phi_rad = azi * np.pi / 180
            sv = self._steering_vec(bs_ant_params['shape'], phi_rad, 0, kd_val)
            codebook_list.append(sv.squeeze())
        F = np.array(codebook_list) # (N_Beams, N_Tx)
        
        # --- Fast Beam Sweeping ---
        best_beams = np.zeros(n_users, dtype=int)
        valid_mask = np.zeros(n_users, dtype=bool)
        
        for i in tqdm(range(n_users), desc=f"Beam Search ({n_beams})"):
            # 1. Prepare Channel
            h_user = channels[i].squeeze() 
            if h_user.ndim == 2:
                h_user = np.mean(h_user, axis=1) # Average over subcarriers for wideband beam
            
            # 2. Beamform (Linear Power) -> |w^H h|^2
            bf_power = np.abs(F @ h_user) ** 2
            
            # 3. Select Best
            best_idx = np.argmax(bf_power)
            
            # Safety check for numerical zeros / deep fades
            if bf_power[best_idx] > 1e-9: 
                best_beams[i] = best_idx
                valid_mask[i] = True
                
        return best_beams, valid_mask
    
    @staticmethod
    def _steering_vec(array, phi, theta, kd):
        idxs = DeepMIMOv3.ant_indices(array)
        resp = DeepMIMOv3.array_response(idxs, phi, theta + np.pi/2, kd)
        return resp / np.linalg.norm(resp)

# -----------------------------------------------------------------------------
# FUNCTION: Training Loop
# -----------------------------------------------------------------------------

def train_downstream(
    model: nn.Module,
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    task_config: TaskConfig,
) -> tuple[float, nn.Module, dict, float]:
    """
    Trains a classifier.
    Returns: (current_f1, model, history, total_train_time)
    """
    device = task_config.device
    model.to(device)

    # Initialize Model & Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=task_config.lr)

    # Scheduler 
    lower_milestone = int(0.3*task_config.epochs)
    upper_milestone = int(0.7*task_config.epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lower_milestone, upper_milestone], gamma=0.1)
    
    crit = nn.CrossEntropyLoss()
    
    # History Container
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = -1.0
    best_model_state = copy.deepcopy(model.state_dict())
    total_train_time = 0.0

    # Training Loop
    for epoch in range(task_config.epochs):
        # --- Train Phase ---
        t0_train = time.time()
        model.train()
        train_loss_sum = 0
        train_batches = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = crit(logits, by)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
        
        if device == 'cuda': torch.cuda.synchronize()
        total_train_time += (time.time() - t0_train)

        scheduler.step()
        avg_train_loss = train_loss_sum / train_batches
        history['train_loss'].append(avg_train_loss)
            
        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0
        val_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                
                logits = model(bx)
                loss = crit(logits, by)
                
                val_loss_sum += loss.item()
                val_batches += 1
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(by.cpu().numpy())
        
        # Calculate Epoch Metrics
        avg_val_loss = val_loss_sum / val_batches
        current_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(current_f1)

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    return best_val_f1, model, history, total_train_time

# -----------------------------------------------------------------------------
# FUNCTION: Main Execution Task
# -----------------------------------------------------------------------------

def run_beam_selection_task(experiment_configs: list, task_config: TaskConfig):
    """
    Executes the Beam Prediction benchmark.
    
    Iterates through:
    1. Complexity: Number of Beams (e.g., 32, 64)
    2. Models: Architectures defined in experiment_configs
    3. Data Efficiency: Training ratios
    4. Noise Robustness: SNR levels
    """
    task_name = task_config.task_name
    device = task_config.device
    print(f"Starting Task: {task_name}")
    
    # --- 1. Setup & Logging ---
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(task_config.results_dir, task_name, task_config.scenario_name, time_now)
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results will be saved to: {results_folder}")

    generator = BeamPredictionGenerator(task_config.scenario_name)
    # Load data once, re-label later
    generator.load_raw_data() 

    tsne_results = []
    
    # Log files
    eff_log = os.path.join(results_folder, "data_efficiency_results.csv")
    snr_log = os.path.join(results_folder, "snr_results.csv")
    res_log = os.path.join(results_folder, "resources_results.csv")

    # Initialize CSVs
    with open(eff_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Num_Beams", "Train_Ratio", "Train_Samples", "Model_Name", "F1_Score", "Training_Time"])
    
    with open(snr_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Num_Beams", "Train_Ratio", "SNR_dB", "Model_Name", "F1_Score"])

    with open(res_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Num_Beams", "Model_Name", "MFLOPs", "Params_M", "Encoder_ms", "Head_ms"])

    # --- 2. Complexity Loop (Codebook Size) ---
    for n_beams in task_config.task_complexity:
        print(f"\n--- Generating Labels for {n_beams} Beams ---")
        
        # Generate labels for this specific codebook size
        X_all, y_all, locs = generator.compute_labels_for_beams(n_beams)
        train_dl, val_dl, test_dl = create_dataloaders(X_all, y_all)
        total_samples = len(X_all)

        # Calculate Reference Power from Training Data (Physics Calibration)
        all_train_x = train_dl.dataset.tensors[0]
        ref_power = torch.mean(torch.abs(all_train_x)**2).item()
        print(f"System Ref Power ({n_beams} Beams): {ref_power:.6e}")

        # Save Map
        pd.DataFrame({
            'x': locs[:, 0], 'y': locs[:, 1], 'z': locs[:, 2], 'label': y_all.numpy()
        }).to_csv(os.path.join(results_folder, f"user_map_data_{n_beams}_beams.csv"), index=False)
        
        # --- 3. Model Loop ---
        for config in experiment_configs:
            name = config.name
            config.output_size = n_beams # Update head size
            print(f"Training model: {name}")
            
            # --- 4. Data Efficiency Loop ---
            for ratio in task_config.train_ratios:
                n_train_samples = int(total_samples * ratio)

                # A. Subset
                frac_train_dl = get_subset(train_dl, ratio)

                # B. Train
                model = build_model_from_config(config)
                final_f1, model, _, total_train_time = train_downstream(model, frac_train_dl, val_dl, task_config)

                all_preds = []
                all_targets = []

                model.eval()
                with torch.no_grad():
                    for bx, by in test_dl:
                        bx, by = bx.to(device), by.to(device)
                        
                        logits = model(bx.to(device))
                        
                        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        all_targets.extend(by.cpu().numpy())

                f1 = f1_score(all_targets, all_preds, average='weighted')

                # C. Log Efficiency
                with open(eff_log, 'a', newline='') as f:
                    csv.writer(f).writerow([n_beams, ratio, n_train_samples, name, f"{f1:.4f}", f"{total_train_time:.2f}s"])

                print(f"\tRatio {ratio:.4f} | F1: {f1:.4f}")

                # --- 5. Noise Robustness Loop (SNR) ---
                for snr in task_config.snr_range:
                    # Calculate physics-compliant noise
                    snr_linear = 10 ** (snr / 10.0)
                    noise_power = ref_power / snr_linear

                    all_preds = []
                    all_targets = []

                    model.eval()
                    with torch.no_grad():
                        for bx, by in test_dl:
                            bx, by = bx.to(device), by.to(device)
                            
                            # Apply Fixed Noise
                            bx_noisy = apply_awgn(bx, noise_power)
                            
                            # Pass bx_noisy to model
                            logits = model(bx_noisy.to(device))
                            
                            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                            all_targets.extend(by.cpu().numpy())

                    f1 = f1_score(all_targets, all_preds, average='weighted')

                    # Log Robustness
                    with open(snr_log, 'a', newline='') as f:
                        csv.writer(f).writerow([n_beams, ratio, snr, name, f"{f1:.4f}"])

                    # print(f"\tSNR {snr} | F1: {f1:.4f}")

            # --- 6. Resources metrics (Once per model/beam config) ---
            input_sample = X_all[0:1]
            cost = get_flops_and_params(model, input_sample, device)
            lat = get_latency(model, input_sample, device)

            with open(res_log, 'a', newline='') as f:
                csv.writer(f).writerow([n_beams, name, cost["MFLOPs"], cost["Params_M"],
                                        f"{lat['Encoder_ms']:.4f}", f"{lat['Head_ms']:.4f}"])

            print(f"\tResources: {cost['MFLOPs']:.2f} MFLOPs | {cost['Params_M']:.2f} M Params")
            
            # --- 7. Latent space (t-SNE) ---
            print("\tRunning t-SNE Analysis...")
            features_np, labels_np = extract_features(model, test_dl, device)

            tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
            emb = tsne.fit_transform(features_np)

            tsne_results.append(pd.DataFrame({
                'tsne_1': emb[:, 0],
                'tsne_2': emb[:, 1],
                'label': labels_np,
                'model_name': name,
                'n_beams': n_beams
            }))
            print("\tt-SNE Complete")

    # --- 8. Finalize ---
    if tsne_results:
        tsne_path = os.path.join(results_folder, "tsne_comparison_data.csv")
        pd.concat(tsne_results, ignore_index=True).to_csv(tsne_path, index=False)
        print(f"t-SNE data saved to: {tsne_path}")