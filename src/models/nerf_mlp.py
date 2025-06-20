import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for NeRF"""
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Create frequency bands
        freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
    def forward(self, x):
        """
        Args:
            x: (N, D) input coordinates
        Returns:
            encoded: (N, encoded_dim) positional encoding
        """
        encoded = []
        
        if self.include_input:
            encoded.append(x)
            
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
            
        return torch.cat(encoded, dim=-1)
    
    def get_output_dim(self, input_dim):
        dim = input_dim * (2 * self.num_freqs)
        if self.include_input:
            dim += input_dim
        return dim

class DensityMLP(nn.Module):
    """MLP for density prediction"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
        self.density_layers = nn.Sequential(*layers)
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # Feature output for color prediction
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        features = self.density_layers(x)
        density = self.density_head(features)
        density = F.relu(density)  # Ensure positive density
        
        feature_vec = self.feature_head(features)
        return density, feature_vec

class ColorMLP(nn.Module):
    """MLP for color prediction with view dependence"""
    def __init__(self, feature_dim, dir_dim, hidden_dim=128):
        super().__init__()
        
        self.color_layers = nn.Sequential(
            nn.Linear(feature_dim + dir_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()  # RGB values in [0, 1]
        )
        
    def forward(self, features, view_dirs):
        combined = torch.cat([features, view_dirs], dim=-1)
        return self.color_layers(combined)

class NeRFWithDINO(nn.Module):
    """Enhanced NeRF model with DINO feature integration"""
    def __init__(
        self, 
        pos_freq=10, 
        dir_freq=4, 
        dino_dim=64,
        hidden_dim=256, 
        num_density_layers=8,
        num_color_layers=2,
        skip_connections=[4]
    ):
        super().__init__()
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(pos_freq)
        self.dir_encoder = PositionalEncoding(dir_freq)
        
        # Calculate dimensions
        self.pos_dim = self.pos_encoder.get_output_dim(3)  # 3D positions
        self.dir_dim = self.dir_encoder.get_output_dim(3)  # 3D directions
        self.dino_dim = dino_dim
        
        # DINO feature fusion
        from lora_dino import NeRFDINOFusion
        self.dino_fusion = NeRFDINOFusion(
            pos_dim=self.pos_dim, 
            dino_dim=dino_dim, 
            hidden_dim=hidden_dim
        )
        
        # Density prediction network
        self.density_mlp = DensityMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_density_layers
        )
        
        # Color prediction network
        self.color_mlp = ColorMLP(
            feature_dim=hidden_dim,
            dir_dim=self.dir_dim,
            hidden_dim=hidden_dim // 2
        )
        
        # Skip connections
        self.skip_connections = skip_connections
        
    def forward(self, positions, directions, dino_features):
        """
        Args:
            positions: (N, 3) 3D positions
            directions: (N, 3) view directions
            dino_features: (N, dino_dim) DINO features at positions
            
        Returns:
            rgb: (N, 3) RGB colors
            density: (N, 1) volume density
        """
        # Positional encoding
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)
        
        # Fuse NeRF positional encoding with DINO features
        fused_features = self.dino_fusion(pos_encoded, dino_features)
        
        # Predict density and intermediate features
        density, feature_vec = self.density_mlp(fused_features)
        
        # Predict color with view dependence
        rgb = self.color_mlp(feature_vec, dir_encoded)
        
        return rgb, density

class VolumeRenderer(nn.Module):
    """Volume rendering implementation"""
    def __init__(self):
        super().__init__()
        
    def forward(self, rgb, density, z_vals, rays_d, noise_std=0.0, white_bkgd=False):
        """
        Volume rendering equation implementation
        
        Args:
            rgb: (N_rays, N_samples, 3) RGB values
            density: (N_rays, N_samples, 1) volume density
            z_vals: (N_rays, N_samples) depth values
            rays_d: (N_rays, 3) ray directions
            
        Returns:
            rgb_rendered: (N_rays, 3) rendered RGB
            depth_rendered: (N_rays,) rendered depth
            weights: (N_rays, N_samples) volume weights
        """
        # Compute distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Multiply by ray direction norm to get real world distances
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # Add noise during training for regularization
        if noise_std > 0.0 and self.training:
            noise = torch.randn_like(density) * noise_std
            density = density + noise
            
        # Compute alpha values
        alpha = 1.0 - torch.exp(-F.relu(density) * dists[..., None])
        
        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1, :]), 1.0 - alpha + 1e-10], dim=-2),
            dim=-2
        )[..., :-1, :]
        
        # Compute weights
        weights = alpha * transmittance
        
        # Render RGB
        rgb_rendered = torch.sum(weights * rgb, dim=-2)
        
        # Render depth
        depth_rendered = torch.sum(weights[..., 0] * z_vals, dim=-1)
        
        # Add white background if specified
        if white_bkgd:
            acc_map = torch.sum(weights, dim=-2)
            rgb_rendered = rgb_rendered + (1.0 - acc_map) * 1.0
            
        return rgb_rendered, depth_rendered, weights[..., 0]

class NeRFLoss(nn.Module):
    """Multi-component loss for NeRF training"""
    def __init__(self, rgb_weight=1.0, depth_weight=0.1, regularization_weight=0.01):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.reg_weight = regularization_weight
        
    def forward(self, predictions, targets, weights=None):
        """
        Args:
            predictions: dict with 'rgb', 'depth', 'weights'
            targets: dict with 'rgb', optionally 'depth'
            weights: optional ray weights for importance sampling
        """
        losses = {}
        
        # RGB reconstruction loss
        rgb_loss = F.mse_loss(predictions['rgb'], targets['rgb'])
        losses['rgb'] = rgb_loss
        
        # Depth consistency loss (if available)
        if 'depth' in targets:
            depth_loss = F.l1_loss(predictions['depth'], targets['depth'])
            losses['depth'] = depth_loss
        
        # Regularization: encourage sparsity in volume weights
        if 'weights' in predictions:
            reg_loss = torch.mean(predictions['weights'] ** 2)
            losses['regularization'] = reg_loss
        
        # Combine losses
        total_loss = self.rgb_weight * losses['rgb']
        
        if 'depth' in losses:
            total_loss += self.depth_weight * losses['depth']
            
        if 'regularization' in losses:
            total_loss += self.reg_weight * losses['regularization']
            
        losses['total'] = total_loss
        return losses