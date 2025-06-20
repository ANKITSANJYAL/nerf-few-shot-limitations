import os
import json
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

def load_blender_data(basedir, split="train", img_size=None, half_res=False):
    """
    Loads synthetic Blender dataset from NeRF paper structure.

    Args:
        basedir (str): Path to dataset (e.g., "./data/nerf_synthetic/lego")
        split (str): One of ["train", "val", "test"]
        img_size (int or None): Resize to (img_size, img_size). Overrides half_res if set.
        half_res (bool): If True and img_size is None, halves original resolution and focal length.

    Returns:
        images (torch.Tensor): (N, 3, H, W)
        poses (torch.Tensor): (N, 4, 4)
        hwf (tuple): (H, W, focal)
    """
    with open(os.path.join(basedir, f"transforms_{split}.json"), "r") as f:
        meta = json.load(f)

    images = []
    poses = []

    for frame in meta["frames"]:
        img_path = os.path.join(basedir, frame["file_path"] + ".png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        W_orig, H_orig = img.size

        # Determine image resizing logic
        if img_size:
            resize_dims = (img_size, img_size)
            focal_scale = img_size / W_orig
        elif half_res:
            resize_dims = (H_orig // 2, W_orig // 2)
            focal_scale = 0.5
        else:
            resize_dims = (H_orig, W_orig)
            focal_scale = 1.0

        transform = T.Compose([
            T.Resize(resize_dims, interpolation=Image.LANCZOS),
            T.ToTensor()
        ])
        img = transform(img)
        images.append(img)

        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        poses.append(torch.from_numpy(pose))

    images = torch.stack(images)        # (N, 3, H, W)
    poses = torch.stack(poses)          # (N, 4, 4)
    _, _, H, W = images.shape

    focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"]) * focal_scale

    return images, poses, (H, W, focal)
