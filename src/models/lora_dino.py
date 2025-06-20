import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
import math

class LoRALinear(nn.Module):
    """Efficient LoRA implementation for linear layers"""
    def __init__(self, original_layer, rank=16, alpha=16, dropout=0.1):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
            
        # LoRA matrices
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        original_out = self.original(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x)))
        return original_out + self.scaling * lora_out

class SpatialDINOFeatures(nn.Module):
    """Extracts spatial DINO features with proper coordinate handling"""
    def __init__(self, model_name="facebook/dinov2-base", lora_rank=16, lora_alpha=16):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get model dimensions
        self.patch_size = self.backbone.config.patch_size
        self.embed_dim = self.backbone.config.hidden_size
        
        # Inject LoRA into attention layers
        self._inject_lora(lora_rank, lora_alpha)
        
        # Learnable 2D positional encoding for spatial awareness
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 256, 64))  # 16x16 patches
        
        # Feature projection for NeRF compatibility
        self.feature_proj = nn.Sequential(
            nn.Linear(self.embed_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        
    def _inject_lora(self, rank, alpha):
        """Inject LoRA into attention layers"""
        for layer in self.backbone.encoder.layer:
            attention = layer.attention.attention
            # Apply LoRA to Q, K, V projections
            attention.query = LoRALinear(attention.query, rank=rank, alpha=alpha)
            attention.key = LoRALinear(attention.key, rank=rank, alpha=alpha)
            attention.value = LoRALinear(attention.value, rank=rank, alpha=alpha)
            
    def forward(self, images):
        """
        Args:
            images: List of PIL images or tensor (B, C, H, W)
        Returns:
            features: (B, H_patches, W_patches, feature_dim)
        """
        if isinstance(images, list):
            # Process PIL images
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(next(self.parameters()).device)
        else:
            # Assume preprocessed tensor
            pixel_values = images
            
        # Get DINO features
        outputs = self.backbone(pixel_values=pixel_values)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
        
        B, N_patches, embed_dim = patch_embeddings.shape
        H_patches = W_patches = int(math.sqrt(N_patches))
        
        # Reshape to spatial format
        spatial_features = patch_embeddings.view(B, H_patches, W_patches, embed_dim)
        
        # Add learnable spatial positional encoding
        pos_embed = self.spatial_pos_embed.view(1, H_patches, W_patches, -1)
        pos_embed = pos_embed.expand(B, -1, -1, -1)
        
        # Combine features with positional encoding
        enhanced_features = torch.cat([spatial_features, pos_embed], dim=-1)
        
        # Project to NeRF-compatible dimension
        projected_features = self.feature_proj(enhanced_features)
        
        return projected_features
    
    def sample_features_at_points(self, features, points_2d):
        """
        Sample features at 2D projected points using bilinear interpolation
        
        Args:
            features: (B, H_patches, W_patches, feature_dim)
            points_2d: (N, 2) normalized coordinates in [-1, 1]
            
        Returns:
            sampled_features: (N, feature_dim)
        """
        B, H_patches, W_patches, feature_dim = features.shape
        
        # Reshape features for grid_sample: (B, feature_dim, H_patches, W_patches)
        features_grid = features.permute(0, 3, 1, 2)
        
        # Prepare grid for sampling: (B, N, 1, 2)
        N = points_2d.shape[0]
        grid = points_2d.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1)
        
        # Sample features
        sampled = F.grid_sample(
            features_grid, grid, 
            mode='bilinear', 
            padding_mode='zeros', # 'border' is unsupported on MPS
            align_corners=False
        )
        
        # Reshape: (B, feature_dim, N, 1) -> (B, N, feature_dim) -> (N, feature_dim)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        
        if B == 1:
            sampled = sampled.squeeze(0)
            
        return sampled

class NeRFDINOFusion(nn.Module):
    """Combines NeRF positional encoding with DINO features"""
    def __init__(self, pos_dim, dino_dim, hidden_dim=256):
        super().__init__()
        self.pos_dim = pos_dim
        self.dino_dim = dino_dim
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(pos_dim + dino_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism for adaptive feature weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 2),  # Weight for pos vs dino
            nn.Softmax(dim=-1)
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, pos_encoding, dino_features):
        """
        Args:
            pos_encoding: (N, pos_dim) positional encoding
            dino_features: (N, dino_dim) DINO features
        """
        # Concatenate features
        combined = torch.cat([pos_encoding, dino_features], dim=-1)
        
        # Initial fusion
        fused = self.fusion(combined)
        
        # Compute attention weights
        weights = self.attention(fused)  # (N, 2)
        
        # Apply attention to original features
        pos_weighted = pos_encoding * weights[:, 0:1]
        dino_weighted = dino_features * weights[:, 1:2]
        
        # Final combination
        final_features = self.fusion(torch.cat([pos_weighted, dino_weighted], dim=-1))
        
        return self.output_proj(final_features)