import os
import sys
import torch
import imageio

# Append parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dino_lora import LoRADINO
from models.nerf_model import NeRFMLP
from models.positional_encoding import PositionalEncoding
from models.ray_sampler import get_rays, sample_points_along_rays
from models.volume_renderer import volume_render_radiance
from models.data_loader import load_blender_data

# Setup device
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load models
dino = LoRADINO().to(device).eval()
nerf = NeRFMLP().to(device).eval()
encoder = PositionalEncoding(num_freqs=10, include_input=True)

# Load dataset
images, poses, (H, W, focal) = load_blender_data("data/nerf_synthetic/lego", split="train")

# Make output folder
os.makedirs("outputs", exist_ok=True)

# Batch inference helper
def run_in_batches(model, inputs, batch_size=32768):
    outputs = []
    for i in range(0, inputs.shape[0], batch_size):
        batch = inputs[i:i + batch_size]
        out = model(batch)
        outputs.append(out)
    return torch.cat(outputs, dim=0)

# Inference for first 3 images
N_samples = 64
for idx in range(3):
    print(f"\n--- Processing Image {idx} ---")

    img_tensor = images[idx]  # (3, H, W)
    pose = poses[idx].to(device)

    # Convert to (H, W, 3) for DINO processor
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    with torch.no_grad():
        # 1. Feature extraction
        feat = dino(img).squeeze(0)  # (1, 257, 768) -> (257, 768)
        feat_map = feat[1:].view(16, 16, 768)  # remove CLS token
        print("✅ DINO features:", feat_map.shape)

        # 2. Ray sampling
        rays_o, rays_d = get_rays(H, W, focal, pose)
        pts, z_vals = sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, N_samples=N_samples)
        print("✅ Sampled points:", pts.shape)

        # 3. Positional Encoding
        pts_encoded = encoder(pts.view(-1, 3))
        print("✅ Encoded points:", pts_encoded.shape)

        # 4. NeRF forward pass
        rgb_sigma = run_in_batches(nerf, pts_encoded).view(H, W, N_samples, 4)
        print("✅ NeRF output:", rgb_sigma.shape)

        # 5. Volume Rendering (correct function with 3 args)
        rgb_map = volume_render_radiance(rgb_sigma, z_vals, rays_d)

        # 6. Save Output
        out_img = (rgb_map.cpu().numpy() * 255).astype("uint8")
        out_path = f"outputs/render_{idx}.png"
        imageio.imwrite(out_path, out_img)
        print(f"✅ Saved rendered image to {out_path}")
