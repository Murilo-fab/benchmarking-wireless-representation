"""
Configuration Data Classes.

This module defines the configuration structures for:
1. TaskConfig: Downstream experiment hyperparameters.
2. ModelConfig: Architecture definitions.

It uses a BaseConfig pattern to standardize JSON serialization/deserialization.

Author: Murilo Ferreira Alves Batista - RWTH Aachen/USP
"""

import os
import json
import torch
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Union, List

# -----------------------------------------------------------------------------
# Base Configuration (Inheritance)
# -----------------------------------------------------------------------------

@dataclass
class BaseConfig:
    """Base class providing JSON serialization/deserialization."""
    
    def save_json(self, file_path: str):
        """Serializes the configuration to a JSON file."""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
        print(f"Config saved to {file_path}")

    @classmethod
    def load_json(cls, file_path: str):
        """Factory method to create a config instance from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found at: {file_path}")
            
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

# -----------------------------------------------------------------------------
# 1. Downstream Task Configuration
# -----------------------------------------------------------------------------

@dataclass
class TaskConfig(BaseConfig):
    """
    Configuration for Downstream Tasks (Classification, Beam Prediction, Power Alloc).
    """
    # Identity
    task_name: str
    scenario_name: str = "city_6_miami"

    # Sweeps
    train_ratios: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8])
    snr_range: List[int] = field(default_factory=lambda: [-20, -15, -10, -5, 0, 5, 10, 15, 20])
    task_complexity: Optional[List[int]] = None

    # Hyperparameters
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    results_dir: str = "../results"

    def __post_init__(self):
        self.results_dir = self.results_dir.rstrip("/")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

# -----------------------------------------------------------------------------
# 2. Model Architecture Configuration
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig(BaseConfig):
    """
    Configuration for Neural Network Architectures.
    """
    name: str
    encoder_type: Optional[Literal["AE", "LWM"]] = None
    mode: Literal["raw", "ae", "cls", "channel_emb"] = "raw"
    
    input_size: int = 2048
    output_size: int = 2
    latent_dim: Optional[int] = None
    weights_path: Optional[str] = None
    
    task_type: Literal["classification", "enhanced_classification", "regression"] = "classification"
    fine_tune_layers: Union[List[str], Literal["full"], None] = None

    def __post_init__(self):
        if self.mode == "ae":
            if self.encoder_type != "AE": raise ValueError("Mode 'ae' requires encoder_type='AE'")
            if self.latent_dim is None: raise ValueError("AE requires latent_dim.")
        
        if self.mode in ["cls", "channel_emb"] and self.encoder_type != "LWM":
            raise ValueError(f"Mode '{self.mode}' requires encoder_type='LWM'")
