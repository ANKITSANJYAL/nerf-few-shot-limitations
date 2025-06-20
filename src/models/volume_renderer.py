import torch
import torch.nn.functional as F

def volume_render_radiance(rgb_sigma, z_vals, rays_d, noise_std=0.0):
    """
    Perform volume rendering to produce final RGB image.

    Args:
        rgb_sigma: (H, W, N_samples, 4) → [R, G, B, sigma]
        z_vals: (H, W, N_samples)
        rays_d: (H, W, 3)
        noise_std: float, noise added to sigma for regularization

    Returns:
        rgb_map: (H, W, 3) rendered image
    """
    rgb = rgb_sigma[..., :3]  # (H, W, N_samples, 3)
    sigma = rgb_sigma[..., 3]  # (H, W, N_samples)

    # Calculate deltas (distances between adjacent samples)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Account for ray directions (convert depth distances to real distances)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Optional noise for regularization
    if noise_std > 0.0:
        sigma += noise_std * torch.randn_like(sigma)

    # Compute alpha values
    alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)

    # Compute weights using alpha compositing
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1
    )[..., :-1]
    weights = alpha * transmittance  # (H, W, N_samples)

    # Final rendered image
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (H, W, 3)

    return rgb_map
