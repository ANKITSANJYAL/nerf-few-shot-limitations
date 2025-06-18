import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import imageio
import time
import torch.nn.functional as F
from PIL import Image
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.nerf_mlp import NeRFMLP
from models.positional_encoding import PositionalEncoding
from models.ray_sampler import get_rays, sample_points_along_rays
from models.volume_renderer import volume_render_radiance
from models.data_loader import load_blender_data

# Device setup
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 200
initial_resolution = 16
mid_resolution = 32
final_resolution = 64
N_samples = 32
BATCH_SIZE = 256
DINO_DIM = 768

# DINO feature extractor
dino = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device).eval()
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# NeRF model
nerf = NeRFMLP(pos_dim=63, dino_dim=DINO_DIM).to(device)
posenc = PositionalEncoding(num_freqs=10)
optimizer = optim.Adam(nerf.parameters(), lr=5e-4)

# Load dataset
images, poses, (H, W, focal) = load_blender_data("data/nerf_synthetic/lego", split="train", img_size=128)

# Precompute rays & DINO features
all_data = []
for img_tensor, pose in zip(images[:5], poses[:5]):
    img_tensor = img_tensor.to(device)
    pose = pose.to(device)

    img_dino = transform(img_tensor.cpu()).unsqueeze(0).to(device)
    print("DINO input shape:", img_dino.shape)

    with torch.no_grad():
        feats = dino._process_input(img_dino)  # (1, 196, 768)
        print("ViT feats shape:", feats.shape)

        B, N, C = feats.shape
        spatial_dim = int(N ** 0.5)
        feats = feats.view(B, spatial_dim, spatial_dim, C).squeeze(0)  # (14, 14, 768)

    all_data.append((img_tensor, pose, feats))

# Training loop
for epoch in range(1, epochs + 1):
    start_time = time.time()
    nerf.train()
    total_loss = 0

    # Progressive resolution
    if epoch <= 20:
        H_ds, W_ds = initial_resolution, initial_resolution
    elif epoch <= 60:
        H_ds, W_ds = mid_resolution, mid_resolution
    else:
        H_ds, W_ds = final_resolution, final_resolution

    print(f"Epoch {epoch}/{epochs} - Resolution: {H_ds}x{W_ds}")
    print("------------------------")

    for view_idx, (img_tensor, pose, dino_feat_map) in enumerate(all_data):
        target = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(H_ds, W_ds), mode="bilinear")[0]
        target = target.permute(1, 2, 0).to(device)  # (H, W, 3)

        rays_o, rays_d = get_rays(H_ds, W_ds, focal, pose)
        pts, z_vals = sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, N_samples=N_samples)

        rays_d = rays_d.to(device)
        z_vals = z_vals.to(device)
        pts = pts.to(device)

        pts_flat = pts.view(-1, 3)
        pts_cam = torch.matmul(pts_flat - pose[:3, 3], pose[:3, :3].T)
        x = (pts_cam[:, 0] / pts_cam[:, 2]) * focal + W / 2
        y = (pts_cam[:, 1] / pts_cam[:, 2]) * focal + H / 2

        # Normalize to [-1, 1] for grid_sample
        x_norm = (x / W) * 2 - 1
        y_norm = (y / H) * 2 - 1
        grid = torch.stack((x_norm, y_norm), dim=-1).view(1, -1, 1, 2)

        dino_grid = dino_feat_map.permute(2, 0, 1).unsqueeze(0)  # (1, 768, 14, 14)
        dino_flat = F.grid_sample(dino_grid, grid, align_corners=True, mode="bilinear")  # (1, 768, N, 1)
        dino_flat = dino_flat.squeeze(0).squeeze(-1).T  # (N, 768)

        pts_encoded = posenc(pts_flat)

        rgb_sigma_flat = []
        for i in range(0, pts_encoded.shape[0], BATCH_SIZE):
            batch = pts_encoded[i:i + BATCH_SIZE]
            dino_batch = dino_flat[i:i + BATCH_SIZE]
            rgb_sigma_flat.append(nerf(batch, dino_batch))

        rgb_sigma = torch.cat(rgb_sigma_flat, dim=0).view(H_ds, W_ds, N_samples, 4)

        print(f"View {view_idx + 1:02d}: rgb min/max = {rgb_sigma[..., :3].min().item():.4f}/{rgb_sigma[..., :3].max().item():.4f}, "
              f"sigma min/max = {rgb_sigma[..., 3].min().item():.4f}/{rgb_sigma[..., 3].max().item():.4f}")

        rgb_pred = volume_render_radiance(rgb_sigma, z_vals, rays_d)
        loss = nn.functional.mse_loss(rgb_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    end_time = time.time()
    print(f"[Epoch {epoch:03d}] Avg Loss: {total_loss / len(all_data):.6f} | Time: {end_time - start_time:.2f}s\n")

    if epoch % 10 == 0:
        out_img = (rgb_pred.detach().cpu().numpy() * 255).astype("uint8")
        os.makedirs("outputs/train_dino_spatial", exist_ok=True)
        imageio.imwrite(f"outputs/train_dino_spatial/train_epoch_{epoch}.png", out_img)
        print(f"✅ Saved outputs/train_dino_spatial/train_epoch_{epoch}.png")