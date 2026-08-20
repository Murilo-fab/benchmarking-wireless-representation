"""
Downstream Model Architectures.

This module defines:
1. Task Heads: Classification (Linear), Regression (Softmax), and Enhanced Heads.
2. Wrapper: A unified interface to connect Tokenizers/Encoders (LWM/AE) with Task Heads.
3. Factory: A builder function to instantiate models from config files.

Author: Murilo Ferreira Alves Batista - RWTH Aachen/USP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local Imports
from .encoder import CSIAutoEncoder, lwm

# =============================================================================
# PART 1: TASK HEADS
# =============================================================================

class SimpleClassificationHead(nn.Module):
    """
    Simple linear probe
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.classifier(x)

class ClassificationHead(nn.Module):
    """
    Standard MLP for Classification (Beam Prediction, LoS/NLoS).
    Structure: Linear -> BN -> ReLU -> Dropout -> ...
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [Batch, Features] or [Batch, Seq, Emb] -> Flatten
        x = x.flatten(start_dim=1)
        return self.classifier(x)

class ResBlock1D(nn.Module):
    """A standard 1D Residual Block."""
    def __init__(self, channels, kernel_size=3):
        super(ResBlock1D, self).__init__()
        # padding="same" ensures the number of users (K) remains constant
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection: Add the original input before the final activation
        out += residual
        out = self.gelu(out)
        
        return out

class ResCNNPowerAllocator(nn.Module):
    def __init__(self, input_dim, hidden_channels=256, num_blocks=4):
        """
        Robust Multi-User Power Allocation Model using a Residual CNN.
        
        Args:
            input_dim (int): Dimension of a single user's representation (e.g., AE latent size).
            hidden_channels (int): Number of feature channels inside the ResNet.
            num_blocks (int): Number of residual blocks (controls the receptive field).
        """
        super(ResCNNPowerAllocator, self).__init__()
        
        # 1. Entry Block: Expands the input features to the hidden channel dimension
        # Kernel size 3 starts mixing interference between adjacent users immediately
        self.entry_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )
        
        # 2. Residual Backbone: Deep feature extraction
        self.res_blocks = nn.Sequential(
            *[ResBlock1D(hidden_channels) for _ in range(num_blocks)]
        )
        
        # 3. Power Prediction Head: Pointwise convolution (kernel=1) to map 
        # the high-dimensional features down to exactly 1 power logit per user
        self.exit_conv = nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor of shape (Batch_Size, K_Users, Input_Dim)
            
        Returns:
            torch.Tensor: Normalized power allocation coefficients of shape (Batch_Size, K_Users)
        """
        # Step 1: Reshape for 1D Convolutions
        # PyTorch Conv1d expects (Batch, Channels, Length)
        # Here, Length = K_Users, and Channels = Input_Dim
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)  # Shape becomes: (Batch, Input_Dim, K)
        
        # Step 2: Extract features and build the interference map
        x = self.entry_conv(x) # Shape: (Batch, Hidden_Channels, K)
        x = self.res_blocks(x) # Shape: (Batch, Hidden_Channels, K)
        
        # Step 3: Predict raw power logits
        logits = self.exit_conv(x) # Shape: (Batch, 1, K)
        
        # Remove the channel dimension
        logits = logits.squeeze(1) # Shape: (Batch, K)
        
        # Step 4: Enforce Physical Power Constraint via Softmax
        power_allocation = F.softmax(logits, dim=-1)
        
        return power_allocation

# =============================================================================
# PART 2: UNIFIED WRAPPER
# =============================================================================

class Wrapper(nn.Module):
    """
    Universal Adapter that connects:
    Input -> [Tokenizer] -> [Encoder] -> [Task Head] -> Output
    
    Handles shape mismatches between Single-User (3D) and Multi-User (4D) data.
    """
    def __init__(self, tokenizer=None, encoder=None, task_head=None, mode="raw"):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.task_head = task_head
        self.mode = mode

    def get_features(self, x):
        """
        Extracts latent representations.
        
        Input Support:
          - Single-User: (Batch, Tx, SC)
          - Multi-User:  (Batch, K_Users, Tx, SC)
        
        Logic:
          1. Detects if input is Multi-User (4D).
          2. Flattens Multi-User batch dimension: (B, K, ...) -> (B*K, ...).
          3. Passes through Encoder (LWM or AE).
          4. Reshapes back to (B, K, Features) if needed.
        """
        # 1. Detect Input Mode
        is_multiuser = (x.ndim == 4)
        
        # 2. Conditional Folding
        if is_multiuser:
            B, K = x.shape[0], x.shape[1]
            x_proc = x.flatten(0, 1) # Merge Batch & Users
        else:
            x_proc = x

        # Path A: Baseline (Raw Complex Features)
        if self.mode == "raw":
            # Concat Real/Imag channel-wise
            return torch.cat((x.real, x.imag), dim=-1)

        # Path B: AutoEncoder (AE)
        elif self.mode == "ae":
            if not self.encoder:
                raise ValueError("Mode 'ae' requires an encoder.")
            
            # Forward pass through AE Encoder
            feats = self.encoder(x_proc)
            
            # Reshape back
            if is_multiuser:
                return feats.view(B, K, -1)
            else:
                return feats.unsqueeze(1) # Add sequence dim
        
        # Path C: LWM (Tokenizer + Transformer)
        elif self.tokenizer:
            # 1. Tokenize (Scale -> Patch -> Add CLS)
            tokens = self.tokenizer(x_proc)
            
            # 2. Transformer Encoder
            # Returns: (Batch, Seq_Len, Emb_Dim), Attention_Maps
            embeddings, _ = self.encoder(tokens)
            
            # Mode C1: CLS Token Only (Classification)
            if self.mode == "cls":
                out = embeddings[:, 0:1, :] # Take index 0
                
                if is_multiuser:
                    return out.view(B, K, -1)
                else:
                    return out

            # Mode C2: Full Channel Embeddings (Power Allocation)
            if self.mode == "channel_emb":
                out = embeddings[:, 1:, :] # Skip CLS, keep patches
                
                if is_multiuser:
                    # (B, K, Seq_Len, Emb_Dim)
                    return out.view(B, K, out.shape[1], -1)
                else:
                    return out
            
        raise ValueError(f"Incompatible mode '{self.mode}' configuration.")

    def forward(self, x):
        features = self.get_features(x)
        return self.task_head(features)

# =============================================================================
# PART 3: FACTORY & UTILS
# =============================================================================

def build_model_from_config(config):
    """
    Factory function: Instantiates a complete model pipeline from config.
    """
    # 1. Build Task Head
    head_map = {
        "simple_classification": SimpleClassificationHead,
        "classification": ClassificationHead,
        "regression": ResCNNPowerAllocator,
    }
    
    if config.task_type not in head_map:
        raise ValueError(f"Unknown task_type: {config.task_type}")
        
    task_head = head_map[config.task_type](config.input_size, config.output_size)

    # 2. Build Backbone
    tokenizer, encoder = None, None

    # Case A: Autoencoder
    if config.encoder_type == "AE":
        if not config.weights_path:
            print("Warning: Initializing AE with random weights (No path provided).")
        encoder = CSIAutoEncoder(latent_dim=config.latent_dim)
        if config.weights_path:
            encoder.load_weights(config.weights_path)
            
    # Case B: LWM
    elif config.encoder_type == "LWM":
        # Standard Tokenizer setup for LWM
        tokenizer = Tokenizer(patch_rows=4, patch_cols=4)
        encoder = lwm.from_pretrained(config.weights_path)

    # 3. Fine-tuning Control (Freeze/Unfreeze)
    if encoder:
        # Default: Freeze everything
        for param in encoder.parameters():
            param.requires_grad = False
            
        # Unfreeze specific layers if requested
        unfreeze_layers(encoder, config.fine_tune_layers)

    return Wrapper(tokenizer, encoder, task_head, config.mode)

def unfreeze_layers(model, fine_tune_layers):
    """Helper to unfreeze specific layers based on substring matching."""
    if not fine_tune_layers or not model:
        return
    
    if fine_tune_layers == "full":
        for param in model.parameters():
            param.requires_grad = True
        return
    
    # Check if requested layers actually exist
    available_layers = [name for name, _ in model.named_parameters()]
    
    for layer_req in fine_tune_layers:
        if not any(layer_req in lname for lname in available_layers):
            print(f"Warning: Layer substring '{layer_req}' not found in model.")
        
        for name, param in model.named_parameters():
            if layer_req in name:
                param.requires_grad = True
