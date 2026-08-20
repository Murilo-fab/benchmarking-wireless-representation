"""
Power Allocation (Sum-Rate Maximization) Task.

This script benchmarks Deep Learning models against traditional wireless baselines 
(WMMSE/FP, Zero-Forcing, Equal Power) for the Power Allocation problem.

It evaluates:
1. Sum-Rate Capacity (bps/Hz).
2. Data Efficiency: Performance vs. Training Size.
3. Noise Robustness: Performance vs. SNR (using fixed noise floor physics).
4. Computational Complexity.

Author: Murilo Ferreira Alves Batista - RWTH Aachen/USP
"""

# --- 1. Standard Library Imports ---
import os
import csv
import time
import copy
import random
from datetime import datetime

# --- 2. Third-Party Imports ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm
import DeepMIMOv3

# --- 3. Local Imports ---
from .data_classes import TaskConfig
from .downstream_models import build_model_from_config
from .utils import (
    get_parameters, 
    create_dataloaders, 
    get_flops_and_params, 
    get_latency, 
    get_subset, 
    apply_awgn
)

import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CLASS: DeepMIMO Generator (Power Allocation Variant)
# -----------------------------------------------------------------------------

class DeepMIMOGenerator:
    """
    Handles data generation for Power Allocation.
    
    Features:
    - Generates user groups based on spatial correlation and gain ratio constraints.
    - Ensures valid user pairing for Multi-User MIMO scenarios.
    
    Attributes:
        params (dict): DeepMIMO parameters.
        scale_factor (float): Normalization factor for channel matrices.
    """
    def __init__(self, scenario_name: str = 'city_6_miami', scale_factor: float = 1e6):
        self.scenario_name = scenario_name
        self.scale_factor = scale_factor

        self.params = get_parameters(scenario_name)
        self.n_subcarriers = self.params['OFDM']['subcarriers']
        self.n_ant_bs = self.params['bs_antenna']['shape'][0]

        # Pre-load data once
        self.all_channels, self.num_total_users, self.all_locations = self.get_channels()
        self.user_gains, self.h_spatial = self.get_matrices()
    
    def get_channels(self) -> tuple[np.ndarray, int, np.ndarray]:
        """Loads and cleans raw DeepMIMO data."""
        deepmimo_data = DeepMIMOv3.generate_data(self.params)

        highest_n_users = 0
        # Iterate through all Base Stations in the list
        for bs_idx in range(len(deepmimo_data)):
            # Filter Valid Users (LoS != -1) for each BS
            idxs = np.where(deepmimo_data[bs_idx]['user']['LoS'] != -1)[0]
            
            if len(idxs) > highest_n_users:
                cleaned_data = deepmimo_data[bs_idx]['user']['channel'][idxs]
                cleaned_locs = deepmimo_data[bs_idx]['user']['location'][idxs]
                
                highest_n_users = len(idxs)
        
        # Remove UE antenna dim: (K, 1, Tx, SC) -> (K, Tx, SC)
        # Using axis=1 to strictly target the UE antenna dimension
        all_channels = cleaned_data.squeeze(axis=1) * self.scale_factor
        num_total_users = all_channels.shape[0]

        return all_channels, num_total_users, cleaned_locs
    
    def get_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculates gains and spatial signatures for user selection."""
        # 1. Gains (Average Power)
        user_gains = np.linalg.norm(self.all_channels, axis=(1, 2))**2 / self.n_subcarriers

        # 2. Spatial Signatures (Normalized Center Subcarrier)
        mid_sub = self.n_subcarriers // 2
        h_spatial_raw = self.all_channels[:, :, mid_sub]
        norms = np.linalg.norm(h_spatial_raw, axis=1, keepdims=True)
        h_spatial = h_spatial_raw / (norms + 1e-9)

        return user_gains, h_spatial

    def get_valid_mask_for_user(self, target_user_idx, min_corr, max_corr, max_gain_ratio):
        """Finds users compatible with the target user (Correlation & Gain checks)."""
        # Correlation check
        target_vec = self.h_spatial[target_user_idx]
        corrs = np.abs(self.h_spatial @ target_vec.conj())
        mask_corr = (corrs >= min_corr) & (corrs <= max_corr)

        # Gain Ratio check
        target_gain = self.user_gains[target_user_idx]
        all_gains = self.user_gains
        g_min = np.minimum(target_gain, all_gains) + 1e-9
        g_max = np.maximum(target_gain, all_gains)
        ratios = g_max / g_min
        mask_gain = ratios <= max_gain_ratio

        return mask_corr & mask_gain
    
    def generate_dataset(self, num_samples, num_users, min_corr=0.3, max_corr=0.9, max_gain_ratio=20.0):
        """
        Greedily samples groups of compatible users.
        Returns:
            dataset_H (Tensor): [Samples, Users, Tx, SC]
        """
        dataset_indices = []
        pbar = tqdm(total=num_samples, desc="Generating User Groups")

        attempts = 0
        while len(dataset_indices) < num_samples:
            # 1. Random Start
            current_group = [np.random.randint(0, self.num_total_users - 1)]
            
            # 2. Init Candidates
            candidate_mask = np.ones(self.num_total_users, dtype=bool)
            candidate_mask[current_group[0]] = False

            group_failed = False
            
            # 3. Greedy Addition
            for _ in range(num_users - 1):
                last_added = current_group[-1]
                step_mask = self.get_valid_mask_for_user(last_added, min_corr, max_corr, max_gain_ratio)
                candidate_mask &= step_mask
                
                valid_indices = np.where(candidate_mask)[0]
                if len(valid_indices) == 0:
                    group_failed = True
                    break
                
                next_user = np.random.choice(valid_indices)
                current_group.append(next_user)
                candidate_mask[next_user] = False
            
            if not group_failed:
                dataset_indices.append(current_group)
                pbar.update(1)
            else:
                attempts += 1
        
        pbar.close()
        indices = np.array(dataset_indices)
        dataset_H = self.all_channels[indices] 
        dataset_gains = self.user_gains[indices]
        dataset_locs = self.all_locations[indices]

        return torch.tensor(dataset_H), dataset_gains, dataset_locs

# -----------------------------------------------------------------------------
# CLASS: Physics-Compliant Loss Function
# -----------------------------------------------------------------------------

class SumRateLoss(nn.Module):
    """
    Differentiable Sum-Rate Calculation for Downlink MU-MIMO.
    Assumes Maximum Ratio Transmission (MRT) Beamforming.
    """
    def __init__(self, noise_power=1e-9):
        super().__init__()
        self.noise_power = noise_power

    def forward(self, pred_powers, channels):
        """
        Args:
            pred_powers (Tensor): [B, N] Normalized power allocation (Sum=1).
            channels (Tensor):    [B, N, Tx, SC] Complex Channel Matrix.
        """
        B, N, Tx, SC = channels.shape
        
        # 1. Power Scaling: Model outputs total power fraction. 
        # Divide by SC because the budget is shared across all subcarriers.
        power_per_sc = pred_powers / SC 
        
        # 2. Vectorization: Merge Batch and Subcarrier dims for parallel computation
        # (B, N, Tx, SC) -> (B*SC, N, Tx)
        channels_folded = channels.permute(0, 3, 1, 2).reshape(B * SC, N, Tx)
        
        # Expand power: (B, N) -> (B*SC, N)
        power_folded = power_per_sc.unsqueeze(1).repeat(1, SC, 1).reshape(B * SC, N)

        # 3. MRT Precoding: w = h* / |h|
        norms = torch.norm(channels_folded, dim=2, keepdim=True) + 1e-12
        w_precoder = torch.conj(channels_folded) / norms 
        
        # Apply Power Amplitude: w_final = w * sqrt(p)
        amplitudes = torch.sqrt(power_folded.unsqueeze(2) + 1e-12)
        w_scaled = w_precoder * amplitudes

        # 4. Effective Channel: H * W^T
        # Result[b, i, j] is signal from Tx-Beam j to User i
        rx_signal_matrix = torch.matmul(channels_folded, w_scaled.transpose(1, 2))
        rx_power_matrix = torch.abs(rx_signal_matrix) ** 2

        # 5. SINR Calculation
        # Signal: Diagonal elements (intended user)
        signal_power = torch.diagonal(rx_power_matrix, dim1=1, dim2=2)
        
        # Interference: Total Power - Signal Power
        total_rx_power = torch.sum(rx_power_matrix, dim=2)
        interference_power = torch.clamp(total_rx_power - signal_power, min=0.0)

        # Shannon Capacity
        sinr = signal_power / (interference_power + self.noise_power + 1e-12)
        rates_per_sc = torch.log2(1 + sinr) 
        
        # 6. Aggregation: Sum over Users and Subcarriers
        sum_rate_per_tone = torch.sum(rates_per_sc, dim=1) 
        sum_rate_matrix = sum_rate_per_tone.view(B, SC)
        total_system_rate = torch.sum(sum_rate_matrix, dim=1)

        # Minimize negative rate
        return -torch.mean(total_system_rate)

# -----------------------------------------------------------------------------
# CLASS: Benchmark Suite (Baselines & SNR Sweep)
# -----------------------------------------------------------------------------

class BenchmarkSuite:
    def __init__(self, snr_levels, device='cpu'):
        self.snr_levels = snr_levels
        self.device = device

    def _calculate_noise_power(self, ref_power, snr_db):
        """Fixed Noise Floor calculation based on Reference Power."""
        snr_linear = 10 ** (snr_db / 10.0)
        return ref_power / snr_linear

    def _compute_raw_rates_mrt(self, powers, channels, noise_power):
        """
        Helper to compute Sum Rate for baselines.
        Rewritten to strictly mirror the physics in SumRateLoss.
        """
        B, N, Tx, SC = channels.shape
        
        # Model outputs total power fraction [B, N]. 
        # Divide by SC because the budget is shared across all subcarriers.
        power_per_sc = powers / SC 
            
        total_rate = torch.zeros(B, device=self.device)

        for k in range(SC):
            H_k = channels[:, :, :, k] # Shape: [B, N, Tx]
            
            # 1. MRT Precoding: w = h* / |h|
            norms = torch.norm(H_k, dim=2, keepdim=True) + 1e-12
            w_precoder = torch.conj(H_k) / norms # Shape: [B, N, Tx]
            
            # 2. Apply Power Amplitude
            p_k = power_per_sc.unsqueeze(-1) # Shape: [B, N, 1]
            amplitudes = torch.sqrt(p_k + 1e-12)
            w_scaled = w_precoder * amplitudes # Shape: [B, N, Tx]
            
            # 3. Effective Channel: H * w_scaled^T
            rx_signal_matrix = torch.matmul(H_k, w_scaled.transpose(1, 2)) # Shape: [B, N, N]
            rx_power_matrix = torch.abs(rx_signal_matrix)**2
            
            # 4. SINR Calculation
            signal_power = torch.diagonal(rx_power_matrix, dim1=1, dim2=2)
            total_rx_power = torch.sum(rx_power_matrix, dim=2)
            interference_power = torch.clamp(total_rx_power - signal_power, min=0.0)
            
            # Shannon Capacity
            sinr = signal_power / (interference_power + noise_power + 1e-12)
            total_rate += torch.sum(torch.log2(1 + sinr), dim=1)
            
        return total_rate
    
    def _compute_zf_rates(self, channels, noise_power):
        """
        Calculates Sum Rate for Zero-Forcing (ZF) Beamforming.
        Strictly enforces the total sum-power constraint per subcarrier.
        """
        B, N, Tx, SC = channels.shape
        total_rate = torch.zeros(B, device=self.device)
        
        # Power budget per subcarrier (Assuming total power budget = 1.0)
        p_budget_per_sc = 1.0 / SC 

        eye = torch.eye(N, dtype=channels.dtype, device=self.device).expand(B, N, N)

        for k in range(SC):
            H_k = channels[:, :, :, k] # Shape: [B, N, Tx]
            
            # 1. Compute Pseudo-Inverse: W_zf = H^H (H H^H)^-1
            Gram = torch.matmul(H_k, H_k.mH)

            Gram_reg = Gram + eye * 1e-9

            X = torch.linalg.solve(Gram_reg, H_k) # Shape: [B, Tx, N]

            W_k = X.mH
            
            # 2. Enforce Total Power Constraint: Tr(W * W^H) = p_budget_per_sc
            # Calculate the current total power of the unnormalized precoder
            current_power = torch.sum(W_k.abs()**2, dim=(1, 2), keepdim=True) # Shape: [B, 1, 1]
            
            # Scale W to meet the exact budget
            scaling_factor = torch.sqrt(p_budget_per_sc / (current_power + 1e-12))
            W_norm = W_k * scaling_factor
            
            # 3. Effective Channel: H * W
            # Because it's ZF, the resulting matrix should be nearly diagonal
            rx_signal_matrix = torch.matmul(H_k, W_norm) # Shape: [B, N, N]
            rx_power_matrix = torch.abs(rx_signal_matrix)**2
            
            # 4. SINR Calculation
            signal_power = torch.diagonal(rx_power_matrix, dim1=1, dim2=2) # Shape: [B, N]
            
            # ZF theoretically zeroes interference, but we calculate it to catch any floating point/normalization artifacts
            total_rx_power = torch.sum(rx_power_matrix, dim=2) # Shape: [B, N]
            interference_power = torch.clamp(total_rx_power - signal_power, min=0.0)
            
            # Shannon Capacity
            sinr = signal_power / (interference_power + noise_power + 1e-12)
            total_rate += torch.sum(torch.log2(1 + sinr), dim=1)
            
        return total_rate

    def compute_baselines(self, test_loader, ref_power, num_restarts=5, pgd_steps=50):
        """Runs Equal, ZF, and Optimal (PGD) baselines."""
        noise_floors = {snr: self._calculate_noise_power(ref_power, snr) for snr in self.snr_levels}
        results = {
            "SNR": self.snr_levels,
            "Equal": {snr: 0.0 for snr in self.snr_levels},
            "ZF": {snr: 0.0 for snr in self.snr_levels},
            "Optimal": {snr: 0.0 for snr in self.snr_levels}
        }
        
        total_samples = 0
        
        for batch in test_loader:
            bx = batch[0].to(self.device)
            B, N = bx.shape[0], bx.shape[1]
            total_samples += B

            p_eq = torch.ones(B, N, device=self.device) / N

            for snr in self.snr_levels:
                current_noise = noise_floors[snr]
                
                # We do not use no_grad here because PGD optimization needs gradients
                # However, for ZF and Eq, we can detach or just compute.
                bx_noisy = apply_awgn(bx, current_noise)
                # 1. Equal
                r_eq = self._compute_raw_rates_mrt(p_eq, bx_noisy, current_noise)
                results["Equal"][snr] += torch.sum(r_eq).item()
                
                # 2. ZF
                r_zf = self._compute_zf_rates(bx_noisy, current_noise)
                results["ZF"][snr] += torch.sum(r_zf).item()
                
                # 3. Optimal (PGD)
                r_opt = self.run_optimization_solver(
                    bx_noisy, steps=pgd_steps, num_restarts=num_restarts,
                    noise_power=current_noise, return_powers=False
                )
                results["Optimal"][snr] += torch.sum(r_opt).item()

        # Average
        final_results = {"SNR": self.snr_levels, "Equal": [], "ZF": [], "Optimal": []}
        for snr in self.snr_levels:
            final_results["Equal"].append(results["Equal"][snr] / total_samples)
            final_results["ZF"].append(results["ZF"][snr] / total_samples)
            final_results["Optimal"].append(results["Optimal"][snr] / total_samples)
            
        return final_results

    def run_model_benchmark(self, model, test_loader, ref_power):
        """Evaluates AI Model robustness against Fixed Noise."""
        model.eval()
        
        noise_floors = {snr: self._calculate_noise_power(ref_power, snr) for snr in self.snr_levels}
        snr_sums = {snr: 0.0 for snr in self.snr_levels}
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                bx_clean = batch[0].to(self.device)
                total_samples += bx_clean.size(0)

                for snr in self.snr_levels:
                    current_noise = noise_floors[snr]

                    # 1. Create Noisy Input (Simulation)
                    bx_noisy = apply_awgn(bx_clean, current_noise)

                    # 2. Inference
                    pred_powers = model(bx_noisy)

                    # 3. Rate Calculation (Real Physics)
                    rates = self._compute_raw_rates_mrt(pred_powers, bx_clean, current_noise)
                    snr_sums[snr] += torch.sum(rates).item()
                    
        ai_results = []
        for snr in self.snr_levels:
            ai_results.append(snr_sums[snr] / total_samples)
            
        return {"SNR": self.snr_levels, "model": ai_results}

    def run_optimization_solver(self, channels, steps=100, num_restarts=20, 
                              noise_power=None, return_powers=False):
        """Projected Gradient Descent (PGD) to find Optimal Power allocation."""
        B, N = channels.shape[0], channels.shape[1]
        if noise_power is None: noise_power = 1e-9

        # Expand for Restarts
        channels_exp = channels.repeat_interleave(num_restarts, dim=0)
        
        # Init Variables
        opt_logits = torch.randn(B * num_restarts, N, device=self.device, requires_grad=True)
        optimizer = optim.Adam([opt_logits], lr=0.1)
        
        for _ in range(steps):
            optimizer.zero_grad()
            p_cand = torch.softmax(opt_logits, dim=1)
            rates = self._compute_raw_rates_mrt(p_cand, channels_exp, noise_power)
            loss = -torch.sum(rates)
            loss.backward()
            optimizer.step()
            
        # Select best restart
        with torch.no_grad():
            final_p = torch.softmax(opt_logits, dim=1)
            all_rates = self._compute_raw_rates_mrt(final_p, channels_exp, noise_power)
            rates_matrix = all_rates.view(B, num_restarts)
            best_rates, best_indices = torch.max(rates_matrix, dim=1)
            
            if return_powers:
                powers_matrix = final_p.view(B, num_restarts, N)
                batch_indices = torch.arange(B, device=self.device)
                return powers_matrix[batch_indices, best_indices, :]
            else:
                return best_rates

# -----------------------------------------------------------------------------
# FUNCTION: Training Loop
# -----------------------------------------------------------------------------

def train_downstream(model, train_loader, val_loader, warmup_loader, task_config, global_noise_power):
    """Trains the Power Allocation model with constant-step scaling and global noise floor."""
    device = task_config.device
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=task_config.lr)

    # We define a fixed number of total steps we want the model to take, regardless of data volume.
    target_total_steps = 5000 
    steps_per_epoch = len(train_loader)
    
    # Calculate how many epochs are needed to hit the exact target steps
    adjusted_epochs = max(10, target_total_steps // steps_per_epoch)
    
    warmup_epochs = int(adjusted_epochs * 0.6)
    self_supervised_epochs = adjusted_epochs - warmup_epochs
    
    total_train_time = 0.0

    # --- Phase 1: Warm-up (Supervised) ---
    mse_criterion = nn.MSELoss()
    
    for epoch in range(warmup_epochs):
        t0 = time.time()
        model.train()

        for bx, target_p in warmup_loader:
            bx, target_p = bx.to(device), target_p.to(device)

            optimizer.zero_grad()
            
            # Model already outputs valid probabilities due to built-in Softmax
            pred_powers = model(bx) 
            
            loss = mse_criterion(pred_powers, target_p)
            loss.backward()
            
            # Prevent early gradient spikes during warm-up
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if device == 'cuda': torch.cuda.synchronize()
        total_train_time += (time.time() - t0)

    # --- Phase 2: Self-Supervised (Sum-Rate Max) ---
    # Lower LR slightly to avoid gradient shock when switching loss landscapes
    for param_group in optimizer.param_groups:
        param_group['lr'] = task_config.lr * 0.2
    
    criterion = SumRateLoss(noise_power=global_noise_power)
    
    scheduler = MultiStepLR(optimizer, milestones=[int(0.3*self_supervised_epochs), int(0.7*self_supervised_epochs)], gamma=0.1)
    
    history = {'train_loss': [], 'val_rate': []}
    best_val_rate = -float('inf')
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(self_supervised_epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        
        for (bx,) in train_loader:
            bx = bx.to(device)

            optimizer.zero_grad()
            pred_powers = model(bx)
            
            # SumRateLoss requires the raw channel to calculate physical interference
            loss = criterion(pred_powers, bx)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if device == 'cuda': torch.cuda.synchronize()
        total_train_time += (time.time() - t0)
        scheduler.step()
        history['train_loss'].append(epoch_loss / len(train_loader))

        # Validate
        model.eval()
        val_rate_sum = 0.0
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                pred_powers = model(bx)
                loss = criterion(pred_powers, bx)
                val_rate_sum += (-loss.item())

        avg_val_rate = val_rate_sum / len(val_loader)
        history['val_rate'].append(avg_val_rate)

        # Early Stopping Checkpointing based strictly on Validation Sum-Rate
        if avg_val_rate > best_val_rate:
            best_val_rate = avg_val_rate
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    return best_val_rate, model, history, total_train_time

def perform_tsne(model, test_loader, device='cpu'):
    """Extracts features and performs t-SNE visualization."""
    model.eval()
    model.to(device)
    features, all_labels = [], []
    max_samples = 2000
    current_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            bx = batch[0].to(device)
            feats = model.get_features(bx)
            preds = model(bx)
            labels_np = torch.argmax(preds, dim=1).cpu().numpy() # Pseudo-labels for coloring
            
            features.append(feats.flatten(start_dim=1))
            all_labels.append(labels_np)
            current_samples += bx.size(0)
            if current_samples >= max_samples: break
            
    X_emb = torch.cat(features, dim=0)[:max_samples].cpu().numpy()
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    return tsne.fit_transform(X_emb), labels

# -----------------------------------------------------------------------------
# FUNCTION: Main Execution Task
# -----------------------------------------------------------------------------

def run_power_allocation_task(experiment_configs: list, task_config: TaskConfig):
    task_name = task_config.task_name
    device = task_config.device
    print(f"Starting Task: {task_name}")
    
    # --- 1. Setup Logs & Folders ---
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(task_config.results_dir, task_name, task_config.scenario_name, time_now)
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results will be saved to: {results_folder}")

    # Log files
    eff_log = os.path.join(results_folder, "data_efficiency_results.csv")
    snr_log = os.path.join(results_folder, "snr_results.csv")
    res_log = os.path.join(results_folder, "resources_results.csv")

    with open(eff_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Num_Users", "Train_Ratio", "Train_Samples", "Model_Name", "Sum_Rate", "Training_Time"])
    with open(snr_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Num_Users", "Train_Ratio", "SNR_dB", "Model_Name", "Sum_Rate"])
    with open(res_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Num_Users", "Model_Name", "MFLOPs", "Params_M", "Encoder_ms", "Head_ms"])

    # --- 2. Data Generation ---
    print(f"\nGenerating dataset for {task_config.scenario_name}")
    generator = DeepMIMOGenerator(task_config.scenario_name)
    benchmark = BenchmarkSuite(task_config.snr_range, device)
    
    total_samples = 2500
    tsne_results = []

    # --- 3. Complexity Loop (Number of Users) ---
    for num_users in task_config.task_complexity:
        print(f"\n--- Scenario: {num_users} Users ---")
        
        # A. Generate Data
        X_all, gains, locs = generator.generate_dataset(num_samples=total_samples, num_users=num_users)
        
        # Save Map
        pd.DataFrame({'x': locs[:, 0, 0], 'y': locs[:, 0, 1], 'z': locs[:, 0, 2], 'gain': gains[:, 0]})\
          .to_csv(os.path.join(results_folder, f"user_map_data_{num_users}_users.csv"), index=False)

        # B. Split Data
        train_dl, val_dl, test_dl = create_dataloaders(X_all)

        # C. Calibration: Calculate System Reference Power (from Train)
        all_train_x = train_dl.dataset.tensors[0]
        ref_power = torch.mean(torch.abs(all_train_x)**2).item()
        print(f"System Ref Power: {ref_power:.6e}")
        
        # Calculate Global Noise Power once for the entire user complexity loop
        # Used fixed 5.0 dB SNR for training loss stability as originally intended
        target_train_snr_linear = 10 ** (5.0 / 10.0) 
        global_noise_power = ref_power / target_train_snr_linear

        # D. Pre-compute Baselines (Optimal, ZF, Equal)
        print(f"\tPre-computing Baselines...")
        baseline_results = benchmark.compute_baselines(test_dl, ref_power=ref_power)

        with open(snr_log, 'a', newline='') as f:
            for i, snr in enumerate(baseline_results["SNR"]):
                csv.writer(f).writerow([num_users, "N/A", snr, "Optimal", baseline_results["Optimal"][i]])
                csv.writer(f).writerow([num_users, "N/A", snr, "Zero-Forcing", baseline_results["ZF"][i]])
                csv.writer(f).writerow([num_users, "N/A", snr, "Equal-Power", baseline_results["Equal"][i]])

        # E. Generate Warm-up Labels (Teacher Mode) - BATCHED to prevent OOM
        print(f"\tGenerating Warm-up Labels (PGD)...")
        X_warmup_list = [batch[0] for batch in train_dl]
        X_warmup = torch.cat(X_warmup_list)
        
        Y_warmup_list = []
        pgd_batch_size = 128 # Safe batch size to prevent CUDA OOM
        
        for i in tqdm(range(0, len(X_warmup), pgd_batch_size), desc="PGD Warmup"):
            batch_x = X_warmup[i:i+pgd_batch_size].to(device)
            # Use the global noise power to ensure PGD matches the Phase 2 Loss physics exactly
            batch_y = benchmark.run_optimization_solver(
                batch_x, steps=50, num_restarts=10, 
                noise_power=global_noise_power, return_powers=True
            ).cpu()
            Y_warmup_list.append(batch_y)
            
        Y_warmup = torch.cat(Y_warmup_list, dim=0)
        
        warmup_ds = TensorDataset(X_warmup, Y_warmup)
        warmup_dl = DataLoader(warmup_ds, batch_size=32, shuffle=True)

        # --- 4. Model Loop ---
        for config in experiment_configs:
            name = config.name
            original_input_size = config.input_size
            
            # Adjust config for Multi-User
            config.input_size = original_input_size # * num_users
            config.output_size = num_users
            print(f"\nTraining model: {name}")

            # --- 5. Data Efficiency Loop ---
            for ratio in task_config.train_ratios:
                frac_train_dl = get_subset(train_dl, ratio)
                frac_warmup_dl = get_subset(warmup_dl, ratio) # Subset of the warmup data too
                n_samples = len(frac_train_dl.dataset)

                # Train
                model = build_model_from_config(config)
                # Pass global_noise_power into the training loop
                sum_rate, model, _, t_train = train_downstream(
                    model, frac_train_dl, val_dl, frac_warmup_dl, task_config, global_noise_power
                )

                # Log
                with open(eff_log, 'a', newline='') as f:
                    csv.writer(f).writerow([num_users, ratio, n_samples, name, f"{sum_rate:.4f}", f"{t_train:.2f}s"])
                
                print(f"\tRatio {ratio:.4f} | Sum Rate: {sum_rate:.4f}")

                # --- 6. Noise Robustness Loop ---
                ai_snr_results = benchmark.run_model_benchmark(model, test_dl, ref_power=ref_power)
                
                for i, snr in enumerate(ai_snr_results["SNR"]):
                    with open(snr_log, 'a', newline='') as f:
                        csv.writer(f).writerow([num_users, ratio, snr, name, ai_snr_results["model"][i]])

            # --- 7. Resources & Visualization ---
            input_sample = X_all[0:1]
            cost = get_flops_and_params(model, input_sample, device)
            lat = get_latency(model, input_sample, device)

            with open(res_log, 'a', newline='') as f:
                csv.writer(f).writerow([num_users, name, cost["MFLOPs"], cost["Params_M"],
                                        f"{lat['Encoder_ms']:.4f}", f"{lat['Head_ms']:.4f}"])

            emb, labels = perform_tsne(model, test_dl, device)
            tsne_results.append(pd.DataFrame({
                'tsne_1': emb[:, 0], 'tsne_2': emb[:, 1],
                'label': labels, 'model_name': name
            }))
            
            # Reset config
            config.input_size = original_input_size

    # Save t-SNE
    if tsne_results:
        pd.concat(tsne_results, ignore_index=True).to_csv(os.path.join(results_folder, "tsne_comparison_data.csv"), index=False)
        print(f"t-SNE data saved.")