"""
Utility Functions for Wireless Channel Deep Learning.

This module provides helper functions for:
1. DeepMIMO Configuration & Data Loading.
2. Physics-based Simulation (AWGN, Channel Noise).
3. Data Splitting & Loader Creation.
4. Model Analysis (FLOPs, Latency, Feature Extraction).

Author: Murilo Ferreira Alves Batista - RWTH Aachen/USP
"""

# --- 1. Standard Library Imports ---
import os
import subprocess
import warnings
from typing import Tuple, Dict, Any, Optional, Union, List

# --- 2. Third-Party Imports ---
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchinfo import summary

# --- 3. Local Imports ---
import DeepMIMOv3

# Suppress specific PyTorch warnings that clutter logs
warnings.filterwarnings("ignore", message="Length of split at index")

# =============================================================================
# PART 1: DEEPMIMO & DATA HANDLING
# =============================================================================

def get_parameters(scenario: str) -> Dict[str, Any]:
    """
    Constructs a robust parameter dictionary for the DeepMIMOv3 engine.
    
    This function:
    1. Loads scenario-specific properties (e.g., max rows, antenna counts).
    2. Sets default physical parameters (32 antennas, 32 subcarriers).
    3. Selects the correct Base Station (BS) index based on the city.
    4. Configures OFDM parameters.

    Args:
        scenario (str): The name of the scenario (e.g., 'city_18_denver').

    Returns:
        dict: A configuration dictionary ready for `DeepMIMOv3.generate_data()`.
    """
    # Constants
    N_ANT = 32
    N_SUB = 32
    SCS = 30e3  # Subcarrier Spacing (Hz)
    DEFAULT_NUM_PATHS = 20

    # 1. Retrieve metadata
    scenario_configs = scenario_prop()
    
    # 2. Initialize defaults
    params = DeepMIMOv3.default_params()
    params['dataset_folder'] = '../scenarios'
    params['scenario'] = scenario.split("_v")[0] # Handle version suffixes

    # 3. BS Selection Logic (Scenario Dependent)
    if scenario in ['city_18_denver', 'city_15_indianapolis']:
        params['active_BS'] = np.array([1, 2, 3])
    else:
        params['active_BS'] = np.array([1])

    # 4. Antenna & Channel Config
    params['enable_BS2BS'] = False
    params['num_paths'] = DEFAULT_NUM_PATHS
    
    params['bs_antenna']['shape'] = np.array([N_ANT, 1]) 
    params['bs_antenna']['rotation'] = np.array([0, 0, -135]) # Standard sector orientation
    params['ue_antenna']['shape'] = np.array([1, 1])          # Single Antenna UE
    
    # 5. User Grid Config
    # Default to 50 rows if not specified in props
    max_rows = scenario_configs.get(scenario, {'n_rows': 50})['n_rows']
    params['user_rows'] = np.arange(max_rows)
    
    # 6. OFDM Config
    params['OFDM']['subcarriers'] = N_SUB
    params['OFDM']['selected_subcarriers'] = np.arange(N_SUB)
    params['OFDM']['bandwidth'] = SCS * N_SUB / 1e9 # GHz
    
    return params

def clone_scenarios(scenario_name: str, repo_url: str, base_dir: str = ".") -> None:
    """
    Clones specific DeepMIMO scenarios using Git Sparse Checkout.
    
    This is bandwidth-efficient: it downloads ONLY the requested scenario folder
    instead of the entire history of all scenarios.

    Args:
        scenario_name (str): Folder name to clone (e.g., 'O1_60').
        repo_url (str): Git repository URL.
        base_dir (str): Local parent directory for the 'scenarios' folder.
    """
    scenarios_path = os.path.join(base_dir, "scenarios")
    if not os.path.exists(scenarios_path):
        os.makedirs(scenarios_path)

    # Initialize Sparse Checkout if new
    if not os.path.exists(os.path.join(scenarios_path, ".git")):
        print(f"Initializing sparse checkout in {scenarios_path}...")
        subprocess.run(["git", "clone", "--sparse", repo_url, "."], cwd=scenarios_path, check=True)
        subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=scenarios_path, check=True)
        subprocess.run(["git", "lfs", "install"], cwd=scenarios_path, check=True)

    # Add requested folder
    print(f"Adding {scenario_name} to sparse checkout...")
    subprocess.run(["git", "sparse-checkout", "add", scenario_name], cwd=scenarios_path, check=True)
    
    # Pull LFS files (large datasets)
    subprocess.run(["git", "lfs", "pull"], cwd=scenarios_path, check=True)
    print(f"Successfully cloned {scenario_name}.")

# =============================================================================
# PART 2: DATA SPLITTING & LOADERS
# =============================================================================

def create_dataloaders(
    inputs: torch.Tensor, 
    labels: Optional[torch.Tensor] = None, 
    train_ratio: float = 0.6, 
    val_ratio: float = 0.2, 
    test_ratio: float = 0.2, 
    batch_size: int = 32, 
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits tensors into Train/Val/Test sets and wraps them in DataLoaders.
    
    Args:
        inputs (Tensor): Input features [N, ...].
        labels (Tensor, optional): Targets [N, ...]. If None, creates TensorDataset(inputs).
        train_ratio (float): Fraction for training (e.g., 0.7).
        val_ratio (float): Fraction for validation.
        test_ratio (float): Fraction for testing.
        batch_size (int): Mini-batch size.
        seed (int): Seed for reproducible random splitting.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    total_samples = len(inputs)
    
    # Sanity Check
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError(f"Ratios sum to {train_ratio + val_ratio + test_ratio:.2f}, must be 1.0")
        
    # Calculate Split Sizes
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = int(total_samples * test_ratio)
    
    # Reproducible Shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(total_samples, generator=g)
    
    # Slicing
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]
    
    # Create Subsets
    x_train, x_val, x_test = inputs[train_idx], inputs[val_idx], inputs[test_idx]
    
    if labels is not None:
        y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
        test_ds = TensorDataset(x_test, y_test)
    else:
        # Unsupervised Case (Autoencoders)
        train_ds = TensorDataset(x_train)
        val_ds = TensorDataset(x_val)
        test_ds = TensorDataset(x_test)
        
    # Create Loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl, test_dl

def get_subset(original_loader: DataLoader, ratio: float, seed: int = 42) -> DataLoader:
    """
    Creates a new DataLoader containing a random subset (x%) of the original data.
    Useful for Data Efficiency experiments (Training on 1%, 10%, etc.).
    
    Args:
        original_loader (DataLoader): The source loader.
        ratio (float): Percentage to keep (0.0 < ratio <= 1.0).
        seed (int): Seed for reproducibility.
        
    Returns:
        DataLoader: A new loader iterating over the subset.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"Ratio must be between 0.0 and 1.0, got {ratio}")

    dataset = original_loader.dataset
    total_samples = len(dataset)
    subset_size = int(total_samples * ratio)
    
    # Random Selection
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(total_samples, generator=g).tolist()
    subset_indices = indices[:subset_size]
    
    # Create Subset
    subset_ds = Subset(dataset, subset_indices)

    # Drop last verification
    batch_size = original_loader.batch_size
    drop_last = (len(subset_ds) % batch_size) == 1
    
    # Preserve original loader settings (workers, pinning)
    new_loader = DataLoader(
        subset_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory,
        drop_last=drop_last
    )
    
    return new_loader

# =============================================================================
# PART 3: PHYSICS & SIMULATION
# =============================================================================

def apply_awgn(x_complex: torch.Tensor, noise_power: float) -> torch.Tensor:
    """
    Applies Complex Gaussian Noise (AWGN) with a fixed variance.
    
    Using fixed noise_power (derived from reference signal power) ensures correct 
    simulation of Path Loss. Far-away users (low signal) naturally get lower SNR 
    than close users (high signal) when noise floor is constant.
    
    Args:
        x_complex (Tensor): Input signal [B, ...].
        noise_power (float): Variance of the noise (N0).
        
    Returns:
        Tensor: Noisy signal.
    """
    # 1. Calculate Standard Deviation
    # Power splits equally into Real and Imaginary parts (P_total = P_real + P_imag)
    # std = sqrt(Power / 2)
    noise_std = torch.sqrt(torch.tensor(noise_power, device=x_complex.device) / 2.0)

    # 2. Generate Noise
    noise_real = torch.randn_like(x_complex.real) * noise_std
    noise_imag = torch.randn_like(x_complex.imag) * noise_std
    
    # 3. Add to Signal
    return x_complex + torch.complex(noise_real, noise_imag)

# =============================================================================
# PART 4: MODEL ANALYSIS & METRICS
# =============================================================================

def extract_features(model: torch.nn.Module, dataloader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs inference to extract latent features (before the final classification head).
    Used for t-SNE visualization.
    
    Args:
        model: Trained PyTorch model.
        dataloader: Data to extract features from.
        device: 'cuda' or 'cpu'.
        
    Returns:
        (features, labels): Numpy arrays of extracted embeddings and corresponding labels.
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            
            # Wrapper models usually expose a 'get_features' method
            if hasattr(model, 'get_features'):
                feats = model.get_features(bx)
                # Flatten spatial dims if CNN (Batch, C, H, W) -> (Batch, Features)
                feats = feats.flatten(start_dim=1)
            else:
                # Fallback: Just run forward pass (if model outputs features directly)
                feats = model(bx)
                
            all_features.append(feats.cpu().numpy())
            all_labels.append(by.cpu().numpy())
            
    return np.concatenate(all_features), np.concatenate(all_labels)

def get_flops_and_params(model: torch.nn.Module, input_tensor: torch.Tensor, device: str) -> Dict[str, float]:
    """
    Calculates theoretical computational cost (FLOPs) and Parameter count.
    
    Args:
        model: PyTorch model.
        input_tensor: A sample input tensor (to determine input shape).
        
    Returns:
        dict: {"MFLOPs": float, "Params_M": float}
    """
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    
    try:
        # torchinfo provides a clean summary
        stats = summary(model, input_data=input_tensor, verbose=0)
        return {
            "MFLOPs": stats.total_mult_adds / 1e6,
            "Params_M": stats.total_params / 1e6
        }
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return {"MFLOPs": 0.0, "Params_M": 0.0}

def get_latency(model: torch.nn.Module, input_tensor: torch.Tensor, device: str, n_repeat: int = 500) -> Dict[str, float]:
    """
    Measures CUDA Latency for the Encoder and the Task Head separately.
    
    Args:
        model: PyTorch wrapper model (must have .get_features() and .task_head()).
        input_tensor: Sample input.
        device: Must be 'cuda'.
        n_repeat: Number of iterations for averaging.
        
    Returns:
        dict: {"Encoder_ms": float, "Head_ms": float}
    """
    if device != "cuda":
        # Latency on CPU is unreliable/variable due to OS scheduling
        return {"Encoder_ms": 0.0, "Head_ms": 0.0}

    input_tensor = input_tensor.to(device)
    
    # 1. Warmup (Wake up GPU)
    with torch.no_grad():
        for _ in range(50):
            if hasattr(model, 'get_features'):
                feats = model.get_features(input_tensor)
                if hasattr(model, 'task_head'):
                    _ = model.task_head(feats)

    # 2. Setup CUDA Events
    start = torch.cuda.Event(enable_timing=True)
    mid   = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    enc_times = []
    head_times = []

    # 3. Measurement Loop
    with torch.no_grad():
        for _ in range(n_repeat):
            start.record()

            # Measure Encoder
            feat = model.get_features(input_tensor) if hasattr(model, 'get_features') else model(input_tensor)

            mid.record()

            # Measure Head (if exists)
            if hasattr(model, 'task_head'):
                _ = model.task_head(feat)

            end.record()

            torch.cuda.synchronize()

            enc_times.append(start.elapsed_time(mid))
            head_times.append(mid.elapsed_time(end))

    return {
        "Encoder_ms": np.mean(enc_times),
        "Head_ms": np.mean(head_times)
    }

def scenario_prop() -> Dict[str, Dict[str, Union[int, List[int]]]]:
    """
    Returns the property dictionary for all supported DeepMIMO scenarios.

    Schema:
    {
        'scenario_name': {
            'n_rows': int | List[int],  # Total rows OR [start_row, end_row] for split datasets
            'n_per_row': int,           # Number of users per row
            'n_ant_bs': int,            # Number of Base Station Antennas
            'n_subcarriers': int        # Number of OFDM Subcarriers
        },
        ...
    }

    Returns:
        dict: A dictionary mapping scenario IDs to their physical configuration.
    """
    row_column_users = {
    'city_0_newyork': {
        'n_rows': 109,
        'n_per_row': 291,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'city_1_losangeles': {
        'n_rows': 142,
        'n_per_row': 201,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'city_2_chicago': {
        'n_rows': 139,
        'n_per_row': 200,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'city_3_houston': {
        'n_rows': 154,
        'n_per_row': 202,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'city_4_phoenix': {
        'n_rows': 198,
        'n_per_row': 214,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'city_5_philadelphia': {
        'n_rows': 239,
        'n_per_row': 164,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'city_6_miami': {
        'n_rows': 199,
        'n_per_row': 216 ,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'city_7_sandiego': {
        'n_rows': 71,
        'n_per_row': 176,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_8_dallas': {
        'n_rows': 207,
        'n_per_row': 190,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'city_9_sanfrancisco': {
        'n_rows': 196,
        'n_per_row': 206,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'city_10_austin': {
        'n_rows': 255,
        'n_per_row': 137,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'city_11_santaclara': {
        'n_rows': 46,
        'n_per_row': 285,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_12_fortworth': {
        'n_rows': 85,
        'n_per_row': 179,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'city_13_columbus': {
        'n_rows': 178,
        'n_per_row': 240,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'city_14_charlotte': {
        'n_rows': 216,
        'n_per_row': 177,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'city_15_indianapolis': {
        'n_rows': 79,
        'n_per_row': 196,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'city_16_sanfrancisco': {
        'n_rows': 201,
        'n_per_row': 208,
        'n_ant_bs': 64,
        'n_subcarriers': 64
    },
    'city_17_seattle': {
        'n_rows': 185,
        'n_per_row': 205,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'city_18_denver': {
        'n_rows': 84,
        'n_per_row': 204,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'city_19_oklahoma': {
        'n_rows': 81,
        'n_per_row': 188,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'asu_campus1_v1': {
        'n_rows': [0, 1*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'asu_campus1_v2': {
        'n_rows': [1*int(321/20), 2*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'asu_campus1_v3': {
        'n_rows': [2*int(321/20), 3*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'asu_campus1_v4': {
        'n_rows': [3*int(321/20), 4*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'asu_campus1_v5': {
        'n_rows': [4*int(321/20), 5*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'asu_campus1_v6': {
        'n_rows': [5*int(321/20), 6*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'asu_campus1_v7': {
        'n_rows': [6*int(321/20), 7*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'asu_campus1_v8': {
        'n_rows': [7*int(321/20), 8*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs':16,
        'n_subcarriers': 64
    },
    'asu_campus1_v9': {
        'n_rows': [8*int(321/20), 9*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'asu_campus1_v10': {
        'n_rows': [9*int(321/20), 10*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'asu_campus1_v11': {
        'n_rows': [10*int(321/20), 11*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'asu_campus1_v12': {
        'n_rows': [11*int(321/20), 12*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'asu_campus1_v13': {
        'n_rows': [12*int(321/20), 13*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'asu_campus1_v14': {
        'n_rows': [13*int(321/20), 14*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'asu_campus1_v15': {
        'n_rows': [14*int(321/20), 15*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'asu_campus1_v16': {
        'n_rows': [15*int(321/20), 16*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'asu_campus1_v17': {
        'n_rows': [16*int(321/20), 17*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 64,
        'n_subcarriers': 64 
    },
    'asu_campus1_v18': {
        'n_rows': [17*int(321/20), 18*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'asu_campus1_v19': {
        'n_rows': [18*int(321/20), 19*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'asu_campus1_v20': {
        'n_rows': [19*int(321/20), 20*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'Boston5G_3p5_v1': {
        'n_rows': [812, 812 + 1*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v2': {
        'n_rows': [812 + 1*int((1622-812)/20), 812 + 2*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'Boston5G_3p5_v3': {
        'n_rows': [812 + 2*int((1622-812)/20), 812 + 3*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'Boston5G_3p5_v4': {
        'n_rows': [812 + 3*int((1622-812)/20), 812 + 4*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'Boston5G_3p5_v5': {
        'n_rows': [812 + 4*int((1622-812)/20), 812 + 5*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'Boston5G_3p5_v6': {
        'n_rows': [812 + 5*int((1622-812)/20), 812 + 6*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'Boston5G_3p5_v7': {
        'n_rows': [812 + 6*int((1622-812)/20), 812 + 7*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v8': {
        'n_rows': [812 + 7*int((1622-812)/20), 812 + 8*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs':16,
        'n_subcarriers': 64
    },
    'Boston5G_3p5_v9': {
        'n_rows': [812 + 8*int((1622-812)/20), 812 + 9*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'Boston5G_3p5_v10': {
        'n_rows': [812 + 9*int((1622-812)/20), 812 + 10*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'Boston5G_3p5_v11': {
        'n_rows': [812 + 10*int((1622-812)/20), 812 + 11*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'Boston5G_3p5_v12': {
        'n_rows': [812 + 11*int((1622-812)/20), 812 + 12*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v13': {
        'n_rows': [812 + 12*int((1622-812)/20), 812 + 13*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'Boston5G_3p5_v14': {
        'n_rows': [812 + 13*int((1622-812)/20), 812 + 14*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'Boston5G_3p5_v15': {
        'n_rows': [812 + 14*int((1622-812)/20), 812 + 15*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'Boston5G_3p5_v16': {
        'n_rows': [812 + 15*int((1622-812)/20), 812 + 16*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v17': {
        'n_rows': [812 + 16*int((1622-812)/20), 812 + 17*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 64,
        'n_subcarriers': 64 
    },
    'Boston5G_3p5_v18': {
        'n_rows': [812 + 17*int((1622-812)/20), 812 + 18*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'Boston5G_3p5_v19': {
        'n_rows': [812 + 18*int((1622-812)/20), 812 + 19*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v20': {
        'n_rows': [812 + 19*int((1622-812)/20), 812 + 20*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'O1_3p5_v1': {
        'n_rows': [0*int(3852/12), 1*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'O1_3p5_v2': {
        'n_rows': [1*int(3852/12), 2*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'O1_3p5_v3': {
        'n_rows': [2*int(3852/12), 3*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'O1_3p5_v4': {
        'n_rows': [3*int(3852/12), 4*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'O1_3p5_v5': {
        'n_rows': [4*int(3852/12), 5*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'O1_3p5_v6': {
        'n_rows': [5*int(3852/12), 6*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'O1_3p5_v7': {
        'n_rows': [6*int(3852/12), 7*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'O1_3p5_v8': {
        'n_rows': [7*int(3852/12), 8*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'O1_3p5_v9': {
        'n_rows': [8*int(3852/12), 9*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'O1_3p5_v10': {
        'n_rows': [9*int(3852/12), 10*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'O1_3p5_v11': {
        'n_rows': [10*int(3852/12), 11*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'O1_3p5_v12': {
        'n_rows': [11*int(3852/12), 12*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'O1_3p5_v13': {
        'n_rows': [12*int(3852/12)+0*int(1351/10), 12*int(3852/12)+1*int(1351/10)],
        'n_per_row': 361,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'O1_3p5_v14': {
        'n_rows': [12*int(3852/12)+1*int(1351/10), 12*int(3852/12)+2*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'O1_3p5_v15': {
        'n_rows': [12*int(3852/12)+2*int(1351/10), 12*int(3852/12)+3*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'O1_3p5_v16': {
        'n_rows': [12*int(3852/12)+3*int(1351/10), 12*int(3852/12)+4*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'O1_3p5_v17': {
        'n_rows': [12*int(3852/12)+4*int(1351/10), 12*int(3852/12)+5*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 64,
        'n_subcarriers': 64
    },
    'O1_3p5_v18': {
        'n_rows': [12*int(3852/12)+5*int(1351/10), 12*int(3852/12)+6*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'O1_3p5_v19': {
        'n_rows': [12*int(3852/12)+6*int(1351/10), 12*int(3852/12)+7*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'O1_3p5_v20': {
        'n_rows': [12*int(3852/12)+7*int(1351/10), 12*int(3852/12)+8*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'city_0_newyork_v16x64': {
        'n_rows': 109,
        'n_per_row': 291,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_1_losangeles_v16x64': {
        'n_rows': 142,
        'n_per_row': 201,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_2_chicago_v16x64': {
        'n_rows': 139,
        'n_per_row': 200,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_3_houston_v16x64': {
        'n_rows': 154,
        'n_per_row': 202,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_4_phoenix_v16x64': {
        'n_rows': 198,
        'n_per_row': 214,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_5_philadelphia_v16x64': {
        'n_rows': 239,
        'n_per_row': 164,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_6_miami_v16x64': {
        'n_rows': 199,
        'n_per_row': 216,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_7_sandiego_v16x64': {
        'n_rows': 207,
        'n_per_row': 176,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_8_dallas_v16x64': {
        'n_rows': 207,
        'n_per_row': 190,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_9_sanfrancisco_v16x64': {
        'n_rows': 196,
        'n_per_row': 206,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    }}
    return row_column_users