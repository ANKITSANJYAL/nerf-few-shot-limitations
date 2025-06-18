import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino_lora import LoRALinear  

class NeRFMLP(nn.Module):
    def __init__(self, pos_dim=63, dino_dim=1024, hidden_dim=256, n_layers=8, lora_rank=16, lora_alpha=16):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            in_dim = pos_dim if i == 0 else hidden_dim
            linear = nn.Linear(in_dim, hidden_dim)

            if i < 2:
                # Apply LoRA to the first 2 layers
                linear = LoRALinear(linear, r=lora_rank, alpha=lora_alpha)

            self.layers.append(linear)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim + dino_dim, hidden_dim),
            nn.ReLU(),
        )

        self.sigma_out = nn.Linear(hidden_dim, 1)
        self.rgb_out = nn.Linear(hidden_dim, 3)

    def forward(self, x, dino_feat=None):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))

        h = torch.cat([h, dino_feat], dim=-1)
        h = self.final_mlp(h)

        sigma = F.softplus(self.sigma_out(h))
        rgb = torch.sigmoid(self.rgb_out(h))
        return torch.cat([rgb, sigma], dim=-1)
