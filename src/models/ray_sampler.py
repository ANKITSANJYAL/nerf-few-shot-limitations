import torch
import torch.nn.functional as F

def get_rays(H, W, focal, c2w):
    """
    Generate rays for all pixels in the image.

    Args:
        H, W: image height and width
        focal: focal length
        c2w: camera-to-world transformation matrix (3x4 or 4x4)

    Returns:
        rays_o: ray origins, shape (H, W, 3)
        rays_d: ray directions, shape (H, W, 3)
    """
    device = c2w.device  # Ensure all tensors are on the same device

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy"
    )
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d

def sample_points_along_rays(rays_o, rays_d, near, far, N_samples, perturb=True):
    """
    Sample 3D points along rays.

    Args:
        rays_o: (H, W, 3) ray origins
        rays_d: (H, W, 3) ray directions
        near, far: float, near and far bounds
        N_samples: int, number of points per ray
        perturb: bool, whether to add randomness

    Returns:
        pts: (H, W, N_samples, 3) sampled points
        z_vals: (H, W, N_samples) depth values
    """
    H, W, _ = rays_o.shape
    device = rays_o.device
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(H, W, N_samples)

    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals
