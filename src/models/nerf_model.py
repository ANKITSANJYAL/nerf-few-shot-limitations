import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRFMLP(nn.Module):
    def __init__(self, pos_dim=63, hidden_dim=256, n_layers=8):  # 🔧 updated pos_dim
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = pos_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))

        self.sigma_out = nn.Linear(hidden_dim, 1)
        self.rgb_out = nn.Linear(hidden_dim, 3)

    def forward(self, x, dir_enc=None):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = F.relu(h)

        sigma = self.sigma_out(h)
        rgb = torch.sigmoid(self.rgb_out(h))
        return torch.cat([rgb, sigma], dim=-1)

