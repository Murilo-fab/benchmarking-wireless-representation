import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# PART 1: LWM MODEL ARCHITECTURE
# Created on Fri Sep 13 19:23:54 2024
# @author: Sadjad Alikhani
# =============================================================================

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class Embedding(nn.Module):
    def __init__(self, element_length, d_model, max_len=513):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.proj = nn.Linear(element_length, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)  
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        seq_len = x.size(1) 
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device) 
        pos_encodings = self.pos_embed(pos)  
        tok_emb = self.proj(x.float()) 
        embedding = tok_emb + pos_encodings 
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = self.scaled_dot_attn(q_s, k_s, v_s)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(output)
        return residual + self.dropout(output), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, enc_inputs):
        # Self-Attention with Add & Norm
        attn_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        attn_outputs = self.norm1(enc_inputs + attn_outputs)  # Add & Norm

        # Feed-Forward with Add & Norm
        ff_outputs = self.pos_ffn(attn_outputs)
        enc_outputs = self.norm2(attn_outputs + ff_outputs)  # Add & Norm

        return enc_outputs, attn


class lwm(nn.Module):
    def __init__(self, element_length=32, d_model=128, n_layers=12, max_len=513, n_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = Embedding(element_length, d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_model*4, dropout) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d_model, d_model)
        self.norm = LayerNormalization(d_model)

        embed_weight = self.embedding.proj.weight
        _, n_dim = embed_weight.size()
        self.decoder = nn.Linear(d_model, n_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_dim))

    @classmethod
    def from_pretrained(cls, ckpt_name='model_weights.pth', device='cuda'):
        model = cls().to(device)
        state_dict = torch.load(ckpt_name, map_location=device)
        
        # Robust Key Cleaning
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
            
        missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
        
        if missing or unexpected:
            print(f"Model loaded. Missing keys: {missing}, Unexpected: {unexpected}")
            
        return model

    def forward(self, input_ids, masked_pos=None):
        # Step 1: Embedding
        output = self.embedding(input_ids)
        attention_maps = []

        # Step 2: Pass through Encoder Layers
        for layer in self.layers:
            output, attn = layer(output)
            attention_maps.append(attn)

        # If masked_pos is provided, perform masked token prediction
        if masked_pos is not None:
            masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(F.relu(self.linear(h_masked))) 
            logits_lm = self.decoder(h_masked) + self.decoder_bias
            return logits_lm, output, attention_maps
        else:
            return output, attention_maps


class RefineBlock(nn.Module):
    """
    A Residual Block that refines features without changing spatial dimensions.
    Used in the Decoder to reconstruct fine details.
    
    Shape: (B, C, H, W) -> (B, C, H, W)
    """
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.net(x)
        return self.relu(out + residual)

class CSIAutoEncoder(nn.Module):
    """
    Convolutional Autoencoder for CSI Compression and Denoising.
    
    Input: Complex CSI (B, Tx, Subcarriers) -> Treated as 2-channel image (Real, Imag).
    Latent: Compressed vector (B, Latent_Dim).
    Output: Reconstructed Complex CSI.
    """
    def __init__(self, latent_dim=64, mode="inference"):
        super().__init__()
        self.mode = mode
        
        # --- Encoder ---
        # Input: (B, 2, 32, 32)
        self.input_norm = nn.BatchNorm2d(2)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (B, 64, 16, 16)
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (B, 128, 8, 8)
            
            nn.Flatten()     # -> (B, 128*8*8) = (B, 8192)
        )
        
        self.flat_dim = 128 * 8 * 8
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        
        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        
        self.decoder_initial = nn.Sequential(
            # Unflatten happens in forward
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.refine1 = nn.Sequential(
            RefineBlock(64),
            RefineBlock(64)
        )
        
        self.decoder_final = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 2, 32, 32)
        )

    def load_weights(self, path, device="cpu"):
        """Safely loads model weights from a .pth file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")
            
        try:
            state_dict = torch.load(path, map_location=device)
            # Handle DataParallel keys
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
            self.load_state_dict(state_dict)
            self.to(device)
            self.eval()
        except RuntimeError as e:
            print(f"Error loading weights: {e}. Check latent_dim config.")
            raise e

    def forward(self, x):
        """
        Args:
            x (Tensor): Complex input [B, Tx, SC]
        Returns:
            If mode='inference': z (Latent Vector)
            If mode='train': (z, x_recon)
        """
        # 1. Preprocess: Complex -> 2-Channel Real
        x_real = torch.stack([x.real, x.imag], dim=1).float() # (B, 2, Tx, SC)
        x_norm = self.input_norm(x_real)
        
        # 2. Encode
        feat = self.encoder(x_norm)
        z = self.fc_enc(feat)
        
        if self.mode == "inference":
            return z
        
        # 3. Decode
        x_recon = self.fc_dec(z).view(-1, 128, 8, 8)
        x_recon = self.decoder_initial(x_recon)
        x_recon = self.refine1(x_recon)
        x_recon = self.decoder_final(x_recon)
        
        # 4. Postprocess: 2-Channel Real -> Complex
        return z, torch.complex(x_recon[:,0], x_recon[:,1])
