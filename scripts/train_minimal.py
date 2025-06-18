import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.nerf_mlp import NeRFMLP
from models.positional_encoding import PositionalEncoding
from models.ray_sampler import get_rays, sample_points_along_rays
from models.volume_renderer import volume_render_radiance
from models.data_loader import load_blender_data

# Device setup
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Hyperparams
epochs = 20
N_samples = 64
BATCH_SIZE = 256

# Load models
nerf = NeRFMLP(pos_dim=63).to(device)
posenc = PositionalEncoding(num_freqs=10)

# Optimizer
optimizer = optim.Adam(nerf.parameters(), lr=5e-4)

# Load dataset (all training views)
images, poses, (H, W, focal) = load_blender_data("data/nerf_synthetic/lego", split="train", img_size=128)

# Precompute all rays and targets
all_rays_o, all_rays_d, all_pts, all_z_vals, all_targets = [], [], [], [], []

def render_in_chunks(rgb_sigma, z_vals, rays_d, chunk_size=64):
    H, W, N, _ = rgb_sigma.shape
    rgb_pred = torch.zeros((H, W, 3), device=rgb_sigma.device)

    for i in range(0, H, chunk_size):
        for j in range(0, W, chunk_size):
            h_end = min(i + chunk_size, H)
            w_end = min(j + chunk_size, W)

            chunk_rgb_sigma = rgb_sigma[i:h_end, j:w_end]
            chunk_z_vals = z_vals[i:h_end, j:w_end]
            chunk_rays_d = rays_d[i:h_end, j:w_end]

            chunk_rgb = volume_render_radiance(chunk_rgb_sigma, chunk_z_vals, chunk_rays_d)
            rgb_pred[i:h_end, j:w_end] = chunk_rgb

    return rgb_pred


for img_tensor, pose in zip(images[:20], poses[:20]):
    img_tensor = img_tensor.to(device)
    pose = pose.to(device)

    target = img_tensor.permute(1, 2, 0).contiguous().reshape(H, W, 3).to(device)
    rays_o, rays_d = get_rays(H, W, focal, pose)
    pts, z_vals = sample_points_along_rays(rays_o, rays_d, 2.0, 6.0, N_samples)

    all_targets.append(target)
    all_rays_o.append(rays_o)
    all_rays_d.append(rays_d)
    all_pts.append(pts)
    all_z_vals.append(z_vals)

num_views = len(all_rays_d)

# Training loop
for epoch in range(1, epochs + 1):
    start_time = time.time()
    nerf.train()
    total_loss = 0

    print(f"Epoch {epoch}/{epochs}")
    print("------------------------")

    for view_idx in range(num_views):
        rays_d = all_rays_d[view_idx].to(device)
        z_vals = all_z_vals[view_idx].to(device)
        pts_flat = all_pts[view_idx].to(device).view(-1, 3)
        rays_d_flat = rays_d.view(-1, 3)
        target = all_targets[view_idx]

        # 1. Positional encoding
        pts_encoded = posenc(pts_flat)

        # 2. Forward pass in batches
        # 2. Forward pass in batches (with on-the-fly positional encoding)
        rgb_sigma_flat = []
        for i in range(0, pts_flat.shape[0], BATCH_SIZE):
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            pts_batch = pts_flat[i:i + BATCH_SIZE]
            encoded_batch = posenc(pts_batch)
            rgb_sigma_flat.append(nerf(encoded_batch))


        rgb_sigma = torch.cat(rgb_sigma_flat, dim=0).view(H, W, N_samples, 4)

        print(f"View {view_idx + 1:02d}: rgb min/max = {rgb_sigma[..., :3].min().item():.4f}/{rgb_sigma[..., :3].max().item():.4f}, "
              f"sigma min/max = {rgb_sigma[..., 3].min().item():.4f}/{rgb_sigma[..., 3].max().item():.4f}")

        # 3. Volume rendering
        rgb_pred = render_in_chunks(rgb_sigma, z_vals, rays_d, chunk_size=64)


        optimizer.zero_grad()

        # 4. Compute loss
        loss = nn.functional.mse_loss(rgb_pred, target)
        
        loss.backward()
        optimizer.step()
        torch.mps.empty_cache()

        total_loss += loss.item()

    end_time = time.time()
    print(f"[Epoch {epoch:03d}] Avg Loss: {total_loss / 20:.6f} | Time: {end_time - start_time:.2f}s\n")

    # Save rendered image for the first view
    if epoch % 5 == 0:
        out_img = (rgb_pred.detach().cpu().numpy() * 255).astype("uint8")
        os.makedirs("outputs/train", exist_ok=True)
        imageio.imwrite(f"outputs/train/train_epoch_{epoch}.png", out_img)
        print(f"✅ Saved outputs/train/train_epoch_{epoch}.png")
