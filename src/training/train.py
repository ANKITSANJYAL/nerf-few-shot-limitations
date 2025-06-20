import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import imageio
import time
from PIL import Image
import matplotlib.pyplot as plt

# Import our modules
from lora_dino import SpatialDINOFeatures
from nerf_mlp import NeRFWithDINO, VolumeRenderer, NeRFLoss
from ray_utils import get_rays, sample_points_along_rays, project_points_to_image, get_ray_batch
from models.data_loader import load_blender_data  # Assume this exists

class NeRFDINOTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize models
        self.dino_model = SpatialDINOFeatures(
            model_name=config['dino_model'],
            lora_rank=config['lora_rank'],
            lora_alpha=config['lora_alpha']
        ).to(self.device)
        
        self.nerf_model = NeRFWithDINO(
            pos_freq=config['pos_freq'],
            dir_freq=config['dir_freq'],
            dino_dim=64,  # Match DINO projection output
            hidden_dim=config['hidden_dim'],
            num_density_layers=config['num_density_layers']
        ).to(self.device)
        
        self.volume_renderer = VolumeRenderer().to(self.device)
        self.criterion = NeRFLoss(
            rgb_weight=config['rgb_weight'],
            depth_weight=config['depth_weight'],
            regularization_weight=config['reg_weight']
        )
        
        # Optimizer - only train LoRA parameters and NeRF
        trainable_params = []
        
        # Add LoRA parameters from DINO
        for name, param in self.dino_model.named_parameters():
            if 'lora' in name.lower() or 'spatial_pos_embed' in name or 'feature_proj' in name:
                trainable_params.append(param)
                print(f"✅ Training DINO param: {name}")
        
        # Add all NeRF parameters
        for name, param in self.nerf_model.named_parameters():
            trainable_params.append(param)
            print(f"✅ Training NeRF param: {name}")
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config['lr_milestones'],
            gamma=config['lr_gamma']
        )
        
        # Training state
        self.epoch = 0
        self.best_psnr = 0.0
        
        # Data preprocessing
        self.image_transform = T.Compose([
            T.Resize((224, 224)),  # DINO input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_data(self, data_path, split='train', max_views=5):
        """Load and preprocess training data"""
        print(f"Loading data from {data_path}...")
        
        # Load images and poses
        images, poses, (H, W, focal) = load_blender_data(
            data_path, split=split, img_size=self.config['img_size']
        )
        
        # Limit number of views for few-shot learning
        images = images[:max_views]
        poses = poses[:max_views]
        
        self.H, self.W, self.focal = H, W, focal
        self.images = [img.permute(1, 2, 0).float() for img in images]
        self.poses = [pose.float() for pose in poses]
        
        print(f"✅ Loaded {len(self.images)} views, resolution: {H}x{W}")
        
        # Precompute DINO features for all views
        self.precompute_dino_features()
        
    def precompute_dino_features(self):
        """Extract DINO features for all training views"""
        print("Precomputing DINO features...")
        
        self.dino_features = []
        self.dino_model.eval()
        
        with torch.no_grad():
            for i, img_tensor in enumerate(self.images):
                # Convert to PIL for DINO processing, ensuring tensor is in (C, H, W) format
                img_pil = T.ToPILImage()(img_tensor.cpu().permute(2, 0, 1))
                
                # Extract spatial DINO features
                features = self.dino_model([img_pil])  # (1, H_patch, W_patch, feature_dim)
                features = features.squeeze(0)  # Remove batch dimension
                
                self.dino_features.append(features)
                print(f"✅ View {i+1}: DINO features shape {features.shape}")
        
        self.dino_model.train()  # Back to training mode
        
    def get_training_rays(self, view_idx, H_sample=None, W_sample=None):
        """Generate training rays for a specific view"""
        H_orig, W_orig = self.H, self.W

        if H_sample is None:
            H_sample = H_orig
        if W_sample is None:
            W_sample = W_orig
            
        # Get rays for the sampled resolution
        rays_o, rays_d = get_rays(H_sample, W_sample, self.focal, self.poses[view_idx])
        
        # Get target image at sampled resolution
        target_img = self.images[view_idx].to(self.device)
        
        # Handle RGBA to RGB conversion
        if target_img.shape[-1] == 4:
            rgb = target_img[..., :3]
            alpha = target_img[..., 3:4]
            target_img = rgb * alpha + (1.0 - alpha) # Blend with white background
        
        if H_sample != H_orig or W_sample != W_orig:
            target_img = torch.nn.functional.interpolate(
                target_img.unsqueeze(0).permute(0, 3, 1, 2),
                size=(H_sample, W_sample),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1).squeeze(0)
        
        return rays_o.to(self.device), rays_d.to(self.device), target_img
    
    def render_rays(self, rays_o, rays_d, view_idx, N_samples=64):
        """Render a batch of rays"""
        # Sample points along rays
        pts, z_vals = sample_points_along_rays(
            rays_o.view(-1, 3), 
            rays_d.view(-1, 3),
            near=self.config['near'],
            far=self.config['far'],
            N_samples=N_samples,
            perturb=self.nerf_model.training
        )
        
        # Flatten points for processing
        N_rays, N_samples_per_ray = pts.shape[:2]
        pts_flat = pts.view(-1, 3)  # (N_rays * N_samples, 3)
        rays_d_flat = rays_d.view(-1, 1, 3).expand(-1, N_samples_per_ray, -1).reshape(-1, 3)
        
        # Project 3D points to image coordinates for DINO feature sampling
        pose = self.poses[view_idx].to(self.device)
        points_2d, depths, valid_mask = project_points_to_image(
            pts_flat, pose, self.focal, self.H, self.W
        )
        
        # Sample DINO features at projected points
        dino_features_view = self.dino_features[view_idx].to(self.device)
        sampled_dino = self.dino_model.sample_features_at_points(
            dino_features_view.unsqueeze(0),  # Add batch dimension
            points_2d
        )
        
        # Handle invalid points (behind camera or outside image)
        if not valid_mask.all():
            # Use mean features for invalid points
            mean_features = sampled_dino.mean(dim=0, keepdim=True)
            sampled_dino = torch.where(
                valid_mask.unsqueeze(-1), 
                sampled_dino, 
                mean_features.expand_as(sampled_dino)
            )
        
        # Forward pass through NeRF
        rgb_flat, density_flat = self.nerf_model(pts_flat, rays_d_flat, sampled_dino)
        
        # Reshape for volume rendering
        rgb = rgb_flat.view(N_rays, N_samples_per_ray, 3)
        density = density_flat.view(N_rays, N_samples_per_ray, 1)
        
        # Volume rendering
        rgb_rendered, depth_rendered, weights = self.volume_renderer(
            rgb, density, z_vals, rays_d.view(-1, 3),
            noise_std=self.config.get('noise_std', 0.0) if self.nerf_model.training else 0.0,
            white_bkgd=self.config.get('white_bkgd', False)
        )
        
        return {
            'rgb': rgb_rendered,
            'depth': depth_rendered,
            'weights': weights
        }
    
    def train_step(self, epoch):
        """Single training step"""
        self.nerf_model.train()
        self.dino_model.train()
        
        # Progressive training schedule
        if epoch < 50:
            H_train, W_train, N_samples = 32, 32, 32
            batch_size = 1024
        elif epoch < 100:
            H_train, W_train, N_samples = 64, 64, 48
            batch_size = 768
        else:
            H_train, W_train, N_samples = 128, 128, 64
            batch_size = 512
        
        total_loss = 0.0
        n_batches = 0
        
        # Train on all views
        for view_idx in range(len(self.images)):
            # Get rays and targets for this view
            rays_o, rays_d, target_rgb = self.get_training_rays(view_idx, H_train, W_train)
            
            # Process in batches to manage memory
            for ray_batch_o, ray_batch_d, pixel_indices in get_ray_batch(rays_o, rays_d, batch_size):
                # Get corresponding target pixels
                target_batch = target_rgb.view(-1, 3)[pixel_indices]
                
                # Render rays
                predictions = self.render_rays(ray_batch_o, ray_batch_d, view_idx, N_samples)
                
                # Compute loss
                targets = {'rgb': target_batch}
                losses = self.criterion(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                losses['total'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.nerf_model.parameters() if p.requires_grad] +
                    [p for p in self.dino_model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                total_loss += losses['total'].item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, view_idx=0):
        """Validation step - render a full image"""
        self.nerf_model.eval()
        self.dino_model.eval()
        
        with torch.no_grad():
            rays_o, rays_d, target_rgb = self.get_training_rays(view_idx)
            
            # Render in chunks to avoid memory issues
            H, W = rays_o.shape[:2]
            rendered_rgb = torch.zeros_like(target_rgb)
            
            chunk_size = 1024
            for ray_batch_o, ray_batch_d, pixel_indices in get_ray_batch(rays_o, rays_d, chunk_size):
                predictions = self.render_rays(ray_batch_o, ray_batch_d, view_idx, N_samples=64)
                
                # Reshape predictions to match pixel indices
                pred_rgb = predictions['rgb']
                rendered_rgb.view(-1, 3)[pixel_indices] = pred_rgb
            
            # Compute PSNR
            mse = torch.mean((rendered_rgb - target_rgb) ** 2)
            psnr = -10.0 * torch.log10(mse)
            
        return rendered_rgb.cpu().numpy(), psnr.item()
    
    def train(self, epochs):
        """Main training loop"""
        print(f"Starting training for {epochs} epochs...")
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training step
            avg_loss = self.train_step(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Validation and logging
            if epoch % self.config['val_freq'] == 0:
                val_img, psnr = self.validate()
                
                # Save validation image
                val_img_uint8 = (np.clip(val_img, 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(self.config['output_dir'], f'val_epoch_{epoch:04d}.png'),
                    val_img_uint8
                )
                
                # Update best model
                if psnr > self.best_psnr:
                    self.best_psnr = psnr
                    self.save_checkpoint(f'best_model.pth')
                
                print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f} | PSNR: {psnr:.2f} | "
                      f"Time: {time.time() - start_time:.2f}s | LR: {self.scheduler.get_last_lr()[0]:.2e}")
            else:
                print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f} | "
                      f"Time: {time.time() - start_time:.2f}s")
            
            # Save checkpoint periodically
            if epoch % self.config['save_freq'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch:04d}.pth')
        
        print(f"✅ Training completed! Best PSNR: {self.best_psnr:.2f}")
    
    def get_final_loss(self):
        """Get the final training loss for analysis"""
        # Run one final training step to get current loss
        self.nerf_model.train()
        self.dino_model.train()
        
        # Use a small batch for final loss computation
        view_idx = 0
        rays_o, rays_d, target_rgb = self.get_training_rays(view_idx, H_sample=32, W_sample=32)
        
        # Process in small batches
        total_loss = 0.0
        n_batches = 0
        
        for ray_batch_o, ray_batch_d, pixel_indices in get_ray_batch(rays_o, rays_d, batch_size=256):
            target_batch = target_rgb.view(-1, 3)[pixel_indices]
            predictions = self.render_rays(ray_batch_o, ray_batch_d, view_idx, N_samples=32)
            targets = {'rgb': target_batch}
            losses = self.criterion(predictions, targets)
            total_loss += losses['total'].item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'nerf_state_dict': self.nerf_model.state_dict(),
            'dino_state_dict': self.dino_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(self.config['output_dir'], filename))
        print(f"💾 Saved checkpoint: {filename}")

def main():
    # Configuration
    config = {
        # Model parameters
        'dino_model': 'facebook/dinov2-base',
        'lora_rank': 16,
        'lora_alpha': 16,
        'pos_freq': 12,
        'dir_freq': 4,
        'hidden_dim': 256,
        'num_density_layers': 10,
        
        # Training parameters
        'learning_rate': 2e-4,
        'weight_decay': 1e-6,
        'lr_milestones': [80, 150],
        'lr_gamma': 0.5,
        'epochs': 200,
        
        # Loss weights
        'rgb_weight': 1.0,
        'depth_weight': 0.1,
        'reg_weight': 0.0001,
        
        # Rendering parameters
        'near': 2.0,
        'far': 6.0,
        'img_size': 128,
        'noise_std': 0.1,
        'white_bkgd': False,
        
        # Logging
        'output_dir': 'outputs/nerf_dino_lora',
        'val_freq': 10,
        'save_freq': 50,
        
        # Data
        'data_path': 'data/nerf_synthetic/lego',
        'max_views': 5
    }
    
    # Initialize trainer
    trainer = NeRFDINOTrainer(config)
    
    # Load data
    trainer.load_data(config['data_path'], max_views=config['max_views'])
    
    # Start training
    trainer.train(config['epochs'])

if __name__ == '__main__':
    main()