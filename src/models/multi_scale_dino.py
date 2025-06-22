import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
import math

class MultiScaleDINOFeatures(nn.Module):
    """Multi-scale DINO feature extraction with hierarchical fusion"""
    
    def __init__(self, model_name="facebook/dinov2-base", use_lora=True, lora_rank=16, lora_alpha=16):
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
        if use_lora:
            self._inject_lora(lora_rank, lora_alpha)
        
        # Multi-scale feature fusion
        self.scales = [1, 2, 4]  # Different downsampling factors
        self.feature_fusion = nn.ModuleDict({
            f'scale_{scale}': nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128)
            ) for scale in self.scales
        })
        
        # Cross-scale attention for feature fusion
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=8, 
            batch_first=True
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(128 * len(self.scales), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        self.output_dim = 128
        
    def _inject_lora(self, rank, alpha):
        """Inject LoRA into attention layers"""
        for layer in self.backbone.encoder.layer:
            attention = layer.attention.attention
            # Apply LoRA to Q, K, V projections
            attention.query = LoRALinear(attention.query, rank=rank, alpha=alpha)
            attention.key = LoRALinear(attention.key, rank=rank, alpha=alpha)
            attention.value = LoRALinear(attention.value, rank=rank, alpha=alpha)
    
    def extract_multi_scale_features(self, images):
        """Extract DINO features at multiple scales"""
        if isinstance(images, list):
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(next(self.parameters()).device)
        else:
            pixel_values = images
            
        B = pixel_values.shape[0]
        multi_scale_features = {}
        
        # Extract features at different scales
        for scale in self.scales:
            # Resize input for this scale
            if scale == 1:
                scaled_input = pixel_values
            else:
                H, W = pixel_values.shape[2], pixel_values.shape[3]
                new_H, new_W = H // scale, W // scale
                scaled_input = F.interpolate(
                    pixel_values, 
                    size=(new_H, new_W), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Get DINO features
            with torch.no_grad():
                outputs = self.backbone(pixel_values=scaled_input)
                patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
            
            # Reshape to spatial format
            N_patches = patch_embeddings.shape[1]
            H_patches = W_patches = int(math.sqrt(N_patches))
            spatial_features = patch_embeddings.view(B, H_patches, W_patches, self.embed_dim)
            
            # Project features for this scale
            projected = self.feature_fusion[f'scale_{scale}'](spatial_features)
            multi_scale_features[scale] = projected
            
        return multi_scale_features
    
    def fuse_multi_scale_features(self, multi_scale_features):
        """Fuse features from different scales using attention"""
        B = next(iter(multi_scale_features.values())).shape[0]
        fused_features = []
        
        # Process each scale
        for scale in self.scales:
            features = multi_scale_features[scale]  # (B, H, W, 128)
            H, W = features.shape[1], features.shape[2]
            
            # Reshape for attention: (B, H*W, 128)
            features_flat = features.view(B, H*W, 128)
            
            # Apply self-attention within this scale
            attended_features, _ = self.cross_scale_attention(
                features_flat, features_flat, features_flat
            )
            
            # Reshape back: (B, H, W, 128)
            attended_features = attended_features.view(B, H, W, 128)
            fused_features.append(attended_features)
        
        # Upsample all features to the same resolution (highest resolution)
        target_H, target_W = fused_features[0].shape[1], fused_features[0].shape[2]
        
        aligned_features = []
        for i, features in enumerate(fused_features):
            if i == 0:  # Already at target resolution
                aligned_features.append(features)
            else:
                # Upsample to target resolution
                upsampled = F.interpolate(
                    features.permute(0, 3, 1, 2),  # (B, 128, H, W)
                    size=(target_H, target_W),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)  # (B, H, W, 128)
                aligned_features.append(upsampled)
        
        # Concatenate all scales
        concatenated = torch.cat(aligned_features, dim=-1)  # (B, H, W, 128*3)
        
        # Final projection
        final_features = self.final_proj(concatenated)  # (B, H, W, 128)
        
        return final_features
    
    def forward(self, images):
        """Extract and fuse multi-scale DINO features"""
        multi_scale_features = self.extract_multi_scale_features(images)
        fused_features = self.fuse_multi_scale_features(multi_scale_features)
        return fused_features
    
    def sample_features_at_points(self, features, points_2d):
        """
        Sample multi-scale features at given 2D points
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
            padding_mode='zeros',  # 'border' is unsupported on MPS
            align_corners=False
        )
        
        # Reshape: (B, feature_dim, N, 1) -> (B, N, feature_dim) -> (N, feature_dim)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        
        if B == 1:
            sampled = sampled.squeeze(0)
            
        return sampled

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