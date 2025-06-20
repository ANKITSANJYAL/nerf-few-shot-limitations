import torch
import numpy as np

def get_rays(H, W, focal, pose):
    """
    Generate rays for all pixels in an image
    
    Args:
        H, W: image height and width
        focal: focal length
        pose: (4, 4) camera-to-world transformation matrix
        
    Returns:
        rays_o: (H, W, 3) ray origins
        rays_d: (H, W, 3) ray directions
    """
    # Create coordinate grid
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='xy'
    )
    
    # Convert to camera coordinates (centered at image center)
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,  # Negative for correct orientation
        -torch.ones_like(i)
    ], dim=-1)
    
    # Transform ray directions to world coordinates
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
    
    # Ray origins are just the camera position
    rays_o = pose[:3, 3].expand(rays_d.shape)
    
    return rays_o, rays_d

def sample_points_along_rays(rays_o, rays_d, near, far, N_samples, perturb=True, lindisp=False):
    """
    Sample points along rays
    
    Args:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        near, far: near and far bounds
        N_samples: number of samples per ray
        perturb: add random perturbation to sample locations
        lindisp: sample linearly in disparity rather than depth
        
    Returns:
        pts: (N_rays, N_samples, 3) 3D points
        z_vals: (N_rays, N_samples) depth values
    """
    N_rays = rays_o.shape[0]
    device = rays_o.device
    
    # Create depth samples
    if lindisp:
        # Sample linearly in inverse depth (disparity)
        t_vals = torch.linspace(0., 1., N_samples, device=device)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        # Sample linearly in depth
        t_vals = torch.linspace(0., 1., N_samples, device=device)
        z_vals = near * (1. - t_vals) + far * t_vals
    
    z_vals = z_vals.expand([N_rays, N_samples])
    
    # Add random perturbation
    if perturb:
        # Get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        
        # Random samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand
    
    # Compute 3D points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    return pts, z_vals

def hierarchical_sampling(rays_o, rays_d, z_vals, weights, N_importance, perturb=True):
    """
    Hierarchical sampling based on coarse model weights
    
    Args:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        z_vals: (N_rays, N_samples) depth values from coarse model
        weights: (N_rays, N_samples) weights from coarse model
        N_importance: number of additional fine samples
        
    Returns:
        pts_fine: (N_rays, N_samples + N_importance, 3) fine sample points
        z_vals_fine: (N_rays, N_samples + N_importance) fine depth values
    """
    N_rays, N_samples = z_vals.shape
    device = z_vals.device
    
    # Add small value to prevent numerical issues
    weights = weights + 1e-5
    
    # Compute PDF
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Take uniform samples
    if perturb:
        u = torch.rand(N_rays, N_importance, device=device)
    else:
        u = torch.linspace(0., 1., N_importance, device=device)
        u = u.expand([N_rays, N_importance])
    
    # Invert CDF using searchsorted
    u = u.contiguous()
    indices = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(indices - 1), indices - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indices), indices)
    indices_g = torch.stack([below, above], dim=-1)
    
    # Gather CDF values
    matched_shape = [N_rays, N_importance, cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indices_g)
    bins_g = torch.gather(z_vals.unsqueeze(1).expand(matched_shape), 2, indices_g)
    
    # Linear interpolation
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    # Combine coarse and fine samples
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, samples], dim=-1), dim=-1)
    
    # Compute 3D points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    
    return pts, z_vals_combined

def get_ray_batch(rays_o, rays_d, batch_size=1024):
    """
    Generate batches of rays for memory-efficient processing
    
    Args:
        rays_o: (H, W, 3) ray origins
        rays_d: (H, W, 3) ray directions
        batch_size: number of rays per batch
        
    Yields:
        batches of (ray_origins, ray_directions, pixel_indices)
    """
    H, W = rays_o.shape[:2]
    N_rays = H * W
    
    # Flatten rays
    rays_o_flat = rays_o.view(-1, 3)
    rays_d_flat = rays_d.view(-1, 3)
    
    # Create pixel indices for reconstruction
    indices = torch.arange(N_rays)
    
    # Generate batches
    for i in range(0, N_rays, batch_size):
        end_i = min(i + batch_size, N_rays)
        yield (
            rays_o_flat[i:end_i],
            rays_d_flat[i:end_i], 
            indices[i:end_i]
        )

def project_points_to_image(points_3d, pose, focal, H, W):
    """
    Project 3D points to image coordinates
    
    Args:
        points_3d: (N, 3) 3D points in world coordinates
        pose: (4, 4) camera-to-world transformation
        focal: focal length
        H, W: image dimensions
        
    Returns:
        points_2d: (N, 2) normalized image coordinates in [-1, 1]
        depths: (N,) depth values
        valid_mask: (N,) boolean mask for points in front of camera
    """
    # Transform to camera coordinates
    pose_inv = torch.inverse(pose)
    points_homo = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim=-1)
    points_cam = torch.matmul(points_homo, pose_inv.T)[..., :3]
    
    # Check if points are in front of camera
    valid_mask = points_cam[..., 2] > 0
    
    # Project to image plane
    x = points_cam[..., 0] / (points_cam[..., 2] + 1e-8) * focal + W / 2
    y = points_cam[..., 1] / (points_cam[..., 2] + 1e-8) * focal + H / 2
    
    # Normalize to [-1, 1] for grid_sample
    x_norm = (x / W) * 2 - 1
    y_norm = (y / H) * 2 - 1
    
    points_2d = torch.stack([x_norm, y_norm], dim=-1)
    depths = points_cam[..., 2]
    
    return points_2d, depths, valid_mask