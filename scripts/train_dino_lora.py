import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import imageio
import time
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.nerf_mlp import NeRFMLP
from models.positional_encoding import PositionalEncoding
from models.ray_sampler import get_rays, sample_points_along_rays
from models.volume_renderer import volume_render_radiance
from models.data_loader import load_blender_data
from models.dino_lora import LoRALinear

# Device setup
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 200
BATCH_SIZE = 256
DINO_DIM = 768
pos_dim = 63

# Helper: Freeze/unfreeze logic
def set_requires_grad(model, requires_grad=True, only_lora=False):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for param in module.parameters():
                param.requires_grad = requires_grad
        elif not only_lora:
            for param in module.parameters():
                param.requires_grad = requires_grad

def print_trainable_params(model):
    print("\n📊 Trainable Parameters:")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  ✅ {name}: {param.shape}")
            count += param.numel()
    print(f"🧠 Total trainable parameters: {count}\n")

# Load frozen DINO ViT
dino = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device).eval()
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Initialize NeRF with LoRA
nerf = NeRFMLP(pos_dim=pos_dim, dino_dim=DINO_DIM, hidden_dim=256, n_layers=8, lora_rank=4).to(device)
posenc = PositionalEncoding(num_freqs=10)

# Start with full MLP + LoRA trainable
set_requires_grad(nerf, True, only_lora=False)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, nerf.parameters()), lr=5e-4)

print("🔓 Phase 1: Full NeRF + LoRA Training")
print_trainable_params(nerf)

# Load 5 training views
images, poses, (H_full, W_full, focal) = load_blender_data("data/nerf_synthetic/lego", split="train", img_size=128)
images, poses = images[:5], poses[:5]

# Precompute DINO features
all_data = []
for img_tensor, pose in zip(images, poses):
    img_tensor = img_tensor.to(device)
    pose = pose.to(device)
    img_dino = transform(img_tensor.cpu()).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dino._process_input(img_dino)  # (1, 196, 768)
        feats = feats.view(1, 14, 14, DINO_DIM).squeeze(0)  # (14, 14, 768)
    all_data.append((img_tensor, pose, feats))

# Training loop
for epoch in range(1, epochs + 1):
    start_time = time.time()
    nerf.train()
    total_loss = 0

    # Freeze MLP after epoch 50
    if epoch == 51:
        print("🔒 Phase 2: LoRA-only Training from Epoch 51")
        set_requires_grad(nerf, False, only_lora=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, nerf.parameters()), lr=5e-4)
        print_trainable_params(nerf)

    # Progressive resolution
    if epoch <= 50:
        H_ds, W_ds, N_samples = 16, 16, 32
    elif epoch <= 100:
        H_ds, W_ds, N_samples = 32, 32, 64
    else:
        H_ds, W_ds, N_samples = 64, 64, 64

    for view_idx, (img_tensor, pose, dino_feat_map) in enumerate(all_data):
        img_tensor = img_tensor.to(device)
        pose = pose.to(device)

        target = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(H_ds, W_ds), mode="bilinear")[0]
        target = target.permute(1, 2, 0)  # (H, W, 3)

        rays_o, rays_d = get_rays(H_ds, W_ds, focal, pose)
        pts, z_vals = sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, N_samples=N_samples)

        rays_d = rays_d.to(device)
        z_vals = z_vals.to(device)
        pts = pts.to(device)
        target = target.to(device)

        pts_flat = pts.view(-1, 3)
        pts_cam = torch.matmul(pts_flat - pose[:3, 3], pose[:3, :3].T)
        x = (pts_cam[:, 0] / pts_cam[:, 2]) * focal + W_full / 2
        y = (pts_cam[:, 1] / pts_cam[:, 2]) * focal + H_full / 2

        x_norm = (x / W_full) * 2 - 1
        y_norm = (y / H_full) * 2 - 1
        grid = torch.stack((x_norm, y_norm), dim=-1).view(1, -1, 1, 2)

        dino_grid = dino_feat_map.permute(2, 0, 1).unsqueeze(0)  # (1, 768, 14, 14)
        dino_flat = torch.nn.functional.grid_sample(dino_grid, grid, align_corners=True, mode="bilinear")
        dino_flat = dino_flat.squeeze(0).squeeze(-1).T  # (N, 768)

        pts_encoded = posenc(pts_flat)

        rgb_sigma_flat = []
        for i in range(0, pts_encoded.shape[0], BATCH_SIZE):
            pe_batch = pts_encoded[i:i + BATCH_SIZE]
            dino_batch = dino_flat[i:i + BATCH_SIZE]
            rgb_sigma_flat.append(nerf(pe_batch, dino_batch))

        rgb_sigma = torch.cat(rgb_sigma_flat, dim=0).view(H_ds, W_ds, N_samples, 4)

        rgb_pred = volume_render_radiance(rgb_sigma, z_vals, rays_d)
        loss = nn.functional.mse_loss(rgb_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    end_time = time.time()
    print(f"[Epoch {epoch:03d}] Loss: {total_loss / len(all_data):.6f} | Time: {end_time - start_time:.2f}s")

    if epoch % 10 == 0:
        out_img = (rgb_pred.detach().cpu().numpy() * 255).astype("uint8")
        os.makedirs("outputs/train_dino_lora", exist_ok=True)
        imageio.imwrite(f"outputs/train_dino_lora/epoch_{epoch:03d}.png", out_img)
        print(f"🖼️ Saved outputs/train_dino_lora/epoch_{epoch:03d}.png")
