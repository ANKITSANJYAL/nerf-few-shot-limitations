import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as T

from models.dino_lora import LoRADINO
from models.nerf_model import NeRFMLP
from models.ray_sampler import get_rays, sample_points_along_rays
from models.positional_encoding import PositionalEncoding
from models.volume_renderer import volume_render_radiance

# -------------------- Config --------------------
H, W = 224, 224
focal = 150.0
near, far = 2.0, 6.0
N_samples = 64
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Load Image ------------------
img_path = "data/sample_chair.png"
image = Image.open(img_path).convert("RGB").resize((W, H))

# ------------------ Extract DINO ----------------
dino = LoRADINO().to(DEVICE)
dino.eval()
with torch.no_grad():
    dino_feats = dino(image).squeeze(0)[1:, :]  # remove CLS, [256, 768]
    dino_feats = dino_feats.view(16, 16, -1)     # spatial grid

# ------------- Ray Generation ------------------
c2w = torch.eye(4)
rays_o, rays_d = get_rays(H, W, focal, c2w)
pts, z_vals = sample_points_along_rays(rays_o, rays_d, near, far, N_samples)

# ------------- Positional Encoding -------------
pts_flat = pts.view(-1, 3)
rays_d_flat = rays_d.view(-1, 3).repeat_interleave(N_samples, dim=0)

pos_enc = PositionalEncoding(num_freqs=10).to(DEVICE)
pts_encoded = pos_enc(pts_flat.to(DEVICE))

# ------------- NeRF Model + Inference ----------
nerf = NeRFMLP(pos_dim=pts_encoded.shape[-1]).to(DEVICE)
nerf.eval()
with torch.no_grad():
    rgb_sigma = nerf(pts_encoded).view(H, W, N_samples, 4)  # [H, W, N, 4]
    rgb_map = volume_render_radiance(rgb_sigma, z_vals.to(DEVICE), rays_d.to(DEVICE))

# ---------------- Visualize --------------------

def run_network_in_chunks(mlp, inputs, chunk_size=65536):
    outputs = []
    for i in range(0, inputs.shape[0], chunk_size):
        chunk = inputs[i:i+chunk_size]
        out = mlp(chunk)
        outputs.append(out)
    return torch.cat(outputs, dim=0)


print("✅ Image loaded:", image.size)
print("✅ DINO features:", dino_feats.shape)  # Should be [16, 16, 768]

print("✅ Rays shape:", rays_o.shape, rays_d.shape)  # [H, W, 3]
print("✅ Sampled points:", pts.shape)  # [H, W, N_samples, 3]
print("✅ z_vals shape:", z_vals.shape)

print("✅ Encoded points:", pts_encoded.shape)  # [H*W*N, encoded_dim]

rgb_sigma = run_network_in_chunks(nerf, pts_encoded).view(H, W, N_samples, 4)
print("✅ NeRF output shape:", rgb_sigma.shape)
print("    rgb min/max:", rgb_sigma[..., :3].min().item(), rgb_sigma[..., :3].max().item())
print("    sigma min/max:", rgb_sigma[..., 3].min().item(), rgb_sigma[..., 3].max().item())
print("✅ Rendering volume...")

rgb_np = rgb_map.cpu().numpy()
plt.imshow(np.clip(rgb_np, 0, 1))
plt.title("Rendered Image from LoRA-NeRF")
plt.axis("off")
plt.show()
