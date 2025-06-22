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
import yaml
import wandb
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from lpips import LPIPS

# Import our modules
from models.dino_feature_model import SpatialDINOFeatures
from models.multi_scale_dino import MultiScaleDINOFeatures
from models.nerf_model import NeRFMLP
from models.nerf_mlp import VolumeRenderer
from models.ray_sampler import sample_points_along_rays
from utils.ray_utils import get_rays, project_points_to_image
from models.data_loader import load_blender_data

class NeRFLoss(nn.Module):
    """A simple NeRF loss module."""
    def __init__(self, rgb_weight, depth_weight, reg_weight):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.reg_weight = reg_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        rgb_pred = predictions['rgb']
        rgb_gt = targets['rgb']
        
        loss_rgb = self.mse_loss(rgb_pred, rgb_gt)
        
        loss_dict = {'rgb': self.rgb_weight * loss_rgb}
        
        return loss_dict

class NeRFDINOTrainer:
    """Main trainer class for our NeRF experiments."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.use_dino = config['model'].get('use_dino', True)
        self.dino_model_type = config['model'].get('dino_model_type', 'single_scale')

        # DINO Feature Model
        if self.use_dino:
            dino_config = config['dino_model']
            if self.dino_model_type == 'multi_scale':
                self.dino_model = MultiScaleDINOFeatures(
                    model_name=dino_config['name'],
                    lora_rank=dino_config['lora_rank'],
                    lora_alpha=dino_config['lora_alpha'],
                    use_lora=dino_config.get('use_lora', True)
                ).to(self.device)
                dino_dim = self.dino_model.output_dim
            else:
                self.dino_model = SpatialDINOFeatures(
                    model_name=dino_config['name'],
                    lora_rank=dino_config['lora_rank'],
                    lora_alpha=dino_config['lora_alpha'],
                    use_lora=dino_config.get('use_lora', True),
                    image_size=self.config['data']['resolution']
                ).to(self.device)
                dino_dim = self.dino_model.output_dim
        else:
            self.dino_model = None
            dino_dim = 0
            
        # NeRF MLP Model
        nerf_config = config['nerf_model']
        self.nerf_model = NeRFMLP(
            pos_freq=nerf_config['pos_freq'],
            dir_freq=nerf_config['dir_freq'],
            hidden_dim=nerf_config['hidden_dim'],
            num_density_layers=nerf_config['num_layers'],
            use_dino=self.use_dino,
            dino_dim=dino_dim
        ).to(self.device)
        
        self.volume_renderer = VolumeRenderer().to(self.device)
        self.criterion = NeRFLoss(
            rgb_weight=config['loss']['rgb_weight'],
            depth_weight=config['loss']['depth_weight'],
            reg_weight=config['loss']['reg_weight']
        )
        
        # Metrics
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.lpips = LPIPS(net='vgg').to(self.device)

        # Optimizer
        trainable_params = list(self.nerf_model.parameters())
        if self.use_dino and config['dino_model'].get('use_lora', True):
            for name, param in self.dino_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
        
        opt_config = config['optimizer']
        self.optimizer = optim.Adam(
            trainable_params,
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=opt_config['lr_milestones'],
            gamma=opt_config['lr_gamma']
        )

        self.epoch = 0
        self.best_psnr = 0.0
        
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_data(self, data_path, split='train', max_views=None):
        print(f"Loading {split} data from {data_path}...")
        
        img_size = self.config['data']['resolution']
        images, poses, (H, W, focal) = load_blender_data(
            data_path, split=split, img_size=img_size
        )
        
        if split == 'train':
            if max_views is not None:
                images = images[:max_views]
                poses = poses[:max_views]
            
            self.H, self.W, self.focal = H, W, focal
            self.images = [img.permute(1, 2, 0).float().to(self.device) for img in images]
            self.poses = [pose.float().to(self.device) for pose in poses]
            print(f"✅ Loaded {len(self.images)} train views, resolution: {H}x{W}")
            if self.use_dino:
                self.precompute_dino_features()
        elif split == 'test':
            self.test_images = [img.permute(1, 2, 0).float().to(self.device) for img in images]
            self.test_poses = [pose.float().to(self.device) for pose in poses]
            print(f"✅ Loaded {len(self.test_images)} test views.")

    def precompute_dino_features(self):
        print("Precomputing DINO features...")
        self.dino_features_precomputed = []
        self.dino_model.eval()
        with torch.no_grad():
            for img in self.images:
                # Use the same transform as during training for consistency
                img_tensor = self.image_transform(img.cpu().numpy()).unsqueeze(0).to(self.device)
                features = self.dino_model(img_tensor)
                self.dino_features_precomputed.append(features)
        self.dino_model.train()
        print("✅ DINO features precomputed.")
        
    def get_rays_for_view(self, view_idx, split='train'):
        if split == 'train':
            pose = self.poses[view_idx].to(self.device)
            target_img = self.images[view_idx].to(self.device)
        else:
            pose = self.test_poses[view_idx].to(self.device)
            target_img = self.test_images[view_idx].to(self.device)
            
        rays_o, rays_d = get_rays(self.H, self.W, self.focal, pose)
        
        if target_img.shape[-1] == 4:
            rgb = target_img[..., :3]
            alpha = target_img[..., 3:4]
            target_img = rgb * alpha + (1.0 - alpha)
            
        return rays_o, rays_d, target_img

    def render_rays(self, rays_o, rays_d, view_idx, N_samples=64):
        pts, z_vals = sample_points_along_rays(
            rays_o.view(-1, 3), 
            rays_d.view(-1, 3),
            self.config['near'],
            self.config['far'],
            N_samples,
            perturb=self.nerf_model.training
        )
        
        N_rays = rays_o.shape[0]
        
        # Get DINO features for the sampled points
        dino_feats = None
        if self.use_dino:
            if self.nerf_model.training:
                # During training, use the corresponding view's DINO map
                feat_idx = view_idx
            else:
                # During evaluation, use the first training view's DINO map
                feat_idx = 0

            # Project 3D points to 2D on the source view
            N_rays, N_samples, _ = pts.shape
            points_2d_normalized, _, _ = project_points_to_image(
                pts.view(-1, 3), self.poses[feat_idx], self.focal, self.H, self.W
            )
            
            # Sample features from the precomputed map
            dino_feats_flat = self.dino_model.sample_features_at_points(
                self.dino_features_precomputed[feat_idx], 
                points_2d_normalized
            )
            dino_feats = dino_feats_flat.view(N_rays, N_samples, -1)

        # Reshape for MLP: (N_rays, N_samples, X) -> (N_rays * N_samples, X)
        pts_flat = pts.view(-1, 3)
        view_dirs_expanded = rays_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)
        dino_feats_flat = dino_feats.view(-1, dino_feats.shape[-1]) if dino_feats is not None else None
        
        # Render with NeRF MLP
        rgb_flat, density_flat = self.nerf_model(pts_flat, view_dirs_expanded, dino_feats_flat)

        # Reshape back to (N_rays, N_samples, X)
        rgb = rgb_flat.view(N_rays, N_samples, 3)
        density = density_flat.view(N_rays, N_samples, 1)

        # Volume rendering
        rgb_rendered, depth_rendered, weights = self.volume_renderer(rgb, density, z_vals, rays_d)
        
        return {
            'rgb': rgb_rendered,
            'depth': depth_rendered,
            'weights': weights
        }

    def train_step(self, epoch):
        self.nerf_model.train()
        if self.use_dino:
            self.dino_model.train()
        
        schedule = self.config['training']['progressive_schedule']
        if epoch < 50:
            H_train, W_train, N_samples = schedule['epochs_0_50']
            batch_size = self.config['training']['batch_size'] * 2
        elif epoch < 100:
            H_train, W_train, N_samples = schedule['epochs_50_100']
            batch_size = self.config['training']['batch_size']
        else:
            H_train, W_train, N_samples = schedule['epochs_100_plus']
            batch_size = self.config['training']['batch_size'] // 2
        
        total_loss = 0.0
        n_batches = 0
        
        for view_idx in range(len(self.images)):
            rays_o_full, rays_d_full, target_rgb_full = self.get_rays_for_view(view_idx, 'train')

            if H_train != self.H or W_train != self.W:
                 focal_scaled = self.focal * (H_train / self.H)
                 pose = self.poses[view_idx].to(self.device)
                 rays_o_full, rays_d_full = get_rays(H_train, W_train, focal_scaled, pose)
                 target_rgb_full = torch.nn.functional.interpolate(target_rgb_full.permute(2,0,1).unsqueeze(0), size=(H_train, W_train), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0)

            ray_indices = torch.randperm(rays_o_full.shape[0] * rays_o_full.shape[1], device=self.device)
            
            for i in range(0, ray_indices.shape[0], batch_size):
                batch_indices = ray_indices[i:i+batch_size]
                ray_batch_o = rays_o_full.view(-1, 3)[batch_indices]
                ray_batch_d = rays_d_full.view(-1, 3)[batch_indices]
                target_batch = target_rgb_full.view(-1, 3)[batch_indices]
                
                predictions = self.render_rays(ray_batch_o, ray_batch_d, view_idx, N_samples)
                
                loss_dict = self.criterion(predictions, {'rgb': target_batch})
                loss = sum(loss_dict.values())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
        return total_loss / n_batches if n_batches > 0 else 0

    @torch.no_grad()
    def evaluate(self, epoch):
        self.nerf_model.eval()
        if self.use_dino:
            self.dino_model.eval()

        all_psnr, all_ssim, all_lpips = [], [], []
        
        output_dir = os.path.join(self.config['output']['save_dir'], f"epoch_{epoch}")
        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(self.test_images)):
            rays_o, rays_d, target_img = self.get_rays_for_view(i, 'test')
            
            rendered_img = []
            chunk_size = self.config['rendering']['chunk_size']
            for j in range(0, rays_o.shape[0] * rays_o.shape[1], chunk_size):
                rays_o_chunk = rays_o.view(-1, 3)[j:j+chunk_size]
                rays_d_chunk = rays_d.view(-1, 3)[j:j+chunk_size]
                
                if rays_o_chunk.shape[0] == 0: continue
                
                preds = self.render_rays(rays_o_chunk, rays_d_chunk, view_idx=0, N_samples=self.config['training']['progressive_schedule']['epochs_100_plus'][2])
                rendered_img.append(preds['rgb'])

            rendered_img = torch.cat(rendered_img, dim=0).view(self.H, self.W, 3)

            gt_permuted = target_img.permute(2, 0, 1).unsqueeze(0)
            rendered_permuted = rendered_img.permute(2, 0, 1).unsqueeze(0)
            
            lpips_gt = gt_permuted * 2 - 1
            lpips_rendered = rendered_permuted * 2 - 1

            all_psnr.append(self.psnr(rendered_permuted, gt_permuted))
            all_ssim.append(self.ssim(rendered_permuted, gt_permuted))
            all_lpips.append(self.lpips(lpips_rendered, lpips_gt))
            
            if i < 5:
                img_to_save = (rendered_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(output_dir, f"render_{i}.png"), img_to_save)
                if epoch == 0:
                    gt_to_save = (target_img.cpu().numpy() * 255).astype(np.uint8)
                    imageio.imwrite(os.path.join(output_dir, f"gt_{i}.png"), gt_to_save)

        mean_psnr = torch.tensor(all_psnr).mean().item()
        mean_ssim = torch.tensor(all_ssim).mean().item()
        mean_lpips = torch.tensor(all_lpips).mean().item()

        return {"psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lpips}

    def train(self, epochs):
        wandb.init(
            project="lora-nerf-few-shot",
            name=self.config['experiment']['name'],
            config=self.config
        )

        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            loss = self.train_step(epoch)
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}")
            wandb.log({"train_loss": loss, "epoch": epoch, "lr": self.scheduler.get_last_lr()[0]})

            if (epoch + 1) % self.config['output']['val_freq'] == 0:
                metrics = self.evaluate(epoch)
                print(f"Validation PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.2f}, LPIPS: {metrics['lpips']:.4f}")
                wandb.log({**metrics, "epoch": epoch})

                if metrics['psnr'] > self.best_psnr:
                    self.best_psnr = metrics['psnr']
                    self.save_checkpoint(f"best_{self.config['experiment']['name']}.pth")
            
            if (epoch + 1) % self.config['output']['save_freq'] == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pth")

        wandb.finish()
        print(f"✅ Training completed! Best PSNR: {self.best_psnr:.2f}")

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.epoch,
            'best_psnr': self.best_psnr,
            'nerf_model_state_dict': self.nerf_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        if self.use_dino:
            checkpoint['dino_model_state_dict'] = self.dino_model.state_dict()
        
        save_path = os.path.join(self.config['output']['save_dir'], filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"💾 Saved checkpoint: {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a NeRF model.")
    parser.add_argument('--config', type=str, required=True, help="Path to config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = NeRFDINOTrainer(config)
    
    data_root = os.path.join("data", config['data']['dataset'], config['data']['scene'])
    trainer.load_data(data_root, 'train', max_views=config['data']['num_views'])
    trainer.load_data(data_root, 'test')
    
    trainer.train(config['training']['epochs'])

if __name__ == '__main__':
    main()