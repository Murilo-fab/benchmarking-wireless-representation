"""
Line-of-Sight (LoS) vs. Non-Line-of-Sight (NLoS) Classification Task.

This script executes a comprehensive benchmark for wireless channel classification.
It evaluates models on three key performance indicators (KPIs):
1. Data Efficiency: Performance vs. Training Set Size.
2. Noise Robustness: Performance vs. SNR (using fixed noise floor physics).
3. Computational Complexity: FLOPs, Parameters, and Latency.

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
import DeepMIMOv3 

# --- 3. Local / Project Imports ---
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
# CLASS: DeepMIMO Generator
# -----------------------------------------------------------------------------

class DeepMIMOGenerator:
    """
    Handles the configuration and generation of wireless channel datasets using DeepMIMOv3.
    
    Attributes:
        params (dict): The DeepMIMO configuration parameters.
        scale_factor (float): Scaling factor to normalize channel values (default: 1e6).
    """
    def __init__(self, scenario_name: str, scale_factor: float = 1e6):
        """
        Initialize the generator.

        Args:
            scenario_name (str): The specific DeepMIMO scenario ID (e.g., 'city_18_denver').
            scale_factor (float): Multiplier for channel values to improve numerical stability.
        """
        self.params = get_parameters(scenario_name)
        self.scale_factor = scale_factor

    def generate_dataset(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Generates the complex channel dataset and LoS labels.

        Process:
        1. Generates raw data via DeepMIMO engine.
        2. Filters users with LoS == -1 (invalid/inactive users).
        3. Squeezes the singleton Receiver Antenna dimension.
        4. Converts to PyTorch Complex64 tensors.

        Returns:
            chs (Tensor): Complex channels [N_Samples, N_Tx, N_Subcarriers].
            labels (Tensor): Binary LoS labels [N_Samples].
            locs (np.ndarray): User locations (x, y, z) [N_Samples, 3].
        
        Raises:
            ValueError: If DeepMIMO generation fails.
        """
        try:
            print(f"Loading DeepMIMO v3 Scenario: {self.params['scenario']}...")
            # DeepMIMOv3 returns a list where each element is a dictionary for one BS
            deepmimo_data = DeepMIMOv3.generate_data(self.params)

            list_chs = []
            list_labels = []
            list_locs = []

            # Iterate through every Base Station (BS) in the simulation
            for bs_idx, bs_entry in enumerate(deepmimo_data):
                # Access the user dictionary for this specific BS
                user_data = bs_entry['user']
                los_labels = user_data['LoS']
                
                # Filter Valid Users (LoS != -1)
                valid_idxs = np.where(los_labels != -1)[0]
                
                if len(valid_idxs) == 0:
                    continue

                # Extract data for valid users
                raw_chs = user_data['channel'][valid_idxs]
                raw_locs = user_data['location'][valid_idxs]
                final_labels = los_labels[valid_idxs]

                # Shape Adjustment: (N, 1_Rx, Tx, Sub) -> (N, Tx, Sub)
                if raw_chs.ndim == 4 and raw_chs.shape[1] == 1:
                    raw_chs = np.squeeze(raw_chs, axis=1)

                list_chs.append(raw_chs)
                list_labels.append(final_labels)
                list_locs.append(raw_locs)

            # 1. Concatenate all BS data along the sample axis (axis 0)
            combined_chs = np.concatenate(list_chs, axis=0)
            combined_labels = np.concatenate(list_labels, axis=0)
            combined_locs = np.concatenate(list_locs, axis=0)

            # 2. Convert to PyTorch Tensors
            chs_tensor = torch.from_numpy(combined_chs).to(torch.complex64) * self.scale_factor
            labels_tensor = torch.from_numpy(combined_labels).to(torch.long)
            
            print(f"Total Samples: {chs_tensor.shape[0]}")
            return chs_tensor, labels_tensor, combined_locs
        
        except Exception as e:
            raise ValueError(f"DeepMIMO Generation Failed: {str(e)}")

# -----------------------------------------------------------------------------
# FUNCTION: Training Loop
# -----------------------------------------------------------------------------

def train_downstream(
    model: nn.Module,
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    task_config: TaskConfig
) -> tuple[float, nn.Module, dict, float]:
    """
    Standard training loop for classification tasks.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data for checkpointing.
        task_config (TaskConfig): Hyperparameters (lr, epochs, device).

    Returns:
        best_val_f1 (float): The best Weighted F1 score achieved.
        model (nn.Module): The model with the best weights loaded.
        history (dict): Logs of training loss and validation metrics.
        total_train_time (float): Total time spent in the training loop (seconds).
    """
    device = task_config.device
    model.to(device)

    # Optimizer Setup
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=task_config.lr)
    crit = nn.CrossEntropyLoss()
    
    # Tracking
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = -1.0
    best_model_state = copy.deepcopy(model.state_dict())
    total_train_time = 0.0
    
    for epoch in range(task_config.epochs):
        # --- Train Phase ---
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = crit(logits, by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Synchronization for accurate timing on GPU
        if device == 'cuda': 
            torch.cuda.synchronize()
        total_train_time += (time.time() - t0)
        
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
            
        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                
                logits = model(bx)
                loss = crit(logits, by)
                val_loss_sum += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(by.cpu().numpy())
        
        # Metrics
        avg_val_loss = val_loss_sum / len(val_loader)
        current_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(current_f1)

        # Checkpointing
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_model_state = copy.deepcopy(model.state_dict())

    # Load best weights before returning
    model.load_state_dict(best_model_state)

    return best_val_f1, model, history, total_train_time

# -----------------------------------------------------------------------------
# FUNCTION: Main Execution Task
# -----------------------------------------------------------------------------

def run_los_nlos_task(experiment_configs: list, task_config: TaskConfig):
    """
    Executes the LoS/NLoS classification benchmark.
    
    Workflow:
    1. Generates Data & Maps.
    2. Calculates System Reference Power (from full training set).
    3. Iterates over Models (Configs).
    4. Iterates over Data Splits (Data Efficiency).
    5. Iterates over SNR Levels (Noise Robustness).
    6. Logs all metrics to CSV.
    """
    task_name = task_config.task_name
    device = task_config.device
    print(f"Starting Task: {task_name}")

    # --- 1. Setup & Logging ---
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(task_config.results_dir, task_name, task_config.scenario_name, time_now)
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results will be saved to: {results_folder}")
    
    # Define Log Files
    eff_log = os.path.join(results_folder, "data_efficiency_results.csv")
    snr_log = os.path.join(results_folder, "snr_results.csv")
    res_log = os.path.join(results_folder, "resources_results.csv")

    # Initialize CSV Headers
    with open(eff_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Train_Ratio", "Train_Samples", "Model_Name", "F1_Score", "Training_Time"])
    with open(snr_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Train_Ratio", "SNR_dB", "Model_Name", "F1_Score"])
    with open(res_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Model_Name", "MFLOPs", "Params_M", "Encoder_ms", "Head_ms"])

    # --- 2. Data Generation ---
    generator = DeepMIMOGenerator(task_config.scenario_name)
    X_all, y_all, locs = generator.generate_dataset()

    # Create base splits
    train_dl, val_dl, test_dl = create_dataloaders(X_all, y_all)

    # Calculate System Reference Power (Physics Calibration)
    # This ensures noise floor is calculated based on the "average" user in the cell.
    all_train_x = train_dl.dataset.tensors[0]
    ref_power = torch.mean(torch.abs(all_train_x)**2).item()
    print(f"System Ref Power (from Train): {ref_power:.6e}")

    # Save User Map
    pd.DataFrame({
        'x': locs[:, 0], 'y': locs[:, 1], 'z': locs[:, 2], 'label': y_all.numpy()
    }).to_csv(os.path.join(results_folder, "user_map_los_nlos.csv"), index=False)
    
    tsne_results = []

    # --- 3. Experiment Loop (Models) ---
    for config in experiment_configs:
        name = config.name
        print(f"\nTraining model: {name}")
        
        # --- 4. Data Efficiency Loop (Splits) ---
        for ratio in task_config.train_ratios:
            
            # A. Prepare Subset
            frac_train_dl = get_subset(train_dl, ratio)
            n_samples = len(frac_train_dl.dataset)

            # B. Train
            model = build_model_from_config(config)
            final_f1, model, _, train_time = train_downstream(model, frac_train_dl, val_dl, task_config)

            # Evaluation
            model.eval()
            all_preds, all_targets = [], []
                
            with torch.no_grad():
                for bx, by in test_dl:
                    bx, by = bx.to(device), by.to(device)
                        
                    logits = model(bx.to(device))
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_targets.extend(by.cpu().numpy())
                
            f1 = f1_score(all_targets, all_preds, average='weighted')

            # C. Log Efficiency
            with open(eff_log, 'a', newline='') as f:
                csv.writer(f).writerow([ratio, n_samples, name, f"{f1:.4f}", f"{train_time:.2f}s"])
            
            print(f"\tRatio {ratio:.4f} ({n_samples} samples) | F1: {f1:.4f}")

            # --- 5. Noise Robustness Loop (SNR) ---
            # We evaluate the CURRENT model (trained on specific ratio) against varying noise.
            for snr in task_config.snr_range:
                
                # Calculate Physics-Compliant Noise Floor
                snr_linear = 10 ** (snr / 10.0)
                noise_power = ref_power / snr_linear

                # Evaluation
                model.eval()
                all_preds, all_targets = [], []
                
                with torch.no_grad():
                    for bx, by in test_dl:
                        bx, by = bx.to(device), by.to(device)
                        
                        # Apply Fixed Noise (Simulates Path Loss effect)
                        bx_noisy = apply_awgn(bx, noise_power)
                        
                        logits = model(bx_noisy.to(device))
                        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        all_targets.extend(by.cpu().numpy())
                
                f1 = f1_score(all_targets, all_preds, average='weighted')

                # Log Robustness
                with open(snr_log, 'a', newline='') as f:
                    csv.writer(f).writerow([ratio, snr, name, f"{f1:.4f}"])
            # --------------------------------------

        # --- 6. Resource Metrics (Once per Model) ---
        # Architecture is constant regardless of data size, so calculate once.
        input_sample = X_all[0:1]
        cost = get_flops_and_params(model, input_sample, device)
        lat = get_latency(model, input_sample, device)

        with open(res_log, 'a', newline='') as f:
            csv.writer(f).writerow([name, cost["MFLOPs"], cost["Params_M"], 
                                    f"{lat['Encoder_ms']:.4f}", f"{lat['Head_ms']:.4f}"])
        
        print(f"\tResources: {cost['MFLOPs']:.2f} MFLOPs | {cost['Params_M']:.2f} M Params")

        # --- 7. Latent Space Visualization (t-SNE) ---
        print("\tRunning t-SNE Analysis...")
        features_np, labels_np = extract_features(model, test_dl, device)
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        emb = tsne.fit_transform(features_np)

        tsne_results.append(pd.DataFrame({
            'tsne_1': emb[:, 0],
            'tsne_2': emb[:, 1],
            'label': labels_np,
            'model_name': name
        }))
        print("\tt-SNE Complete")

    # --- 8. Finalize ---
    if tsne_results:
        tsne_path = os.path.join(results_folder, "tsne_comparison_data.csv")
        pd.concat(tsne_results, ignore_index=True).to_csv(tsne_path, index=False)
        print(f"\nt-SNE data saved to: {tsne_path}")