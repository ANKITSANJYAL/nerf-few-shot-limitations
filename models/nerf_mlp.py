import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRFMLP(nn.Module):
    def __init__(self, pos_dim=63, hidden_dim=256, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = pos_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))

        self.sigma_out = nn.Linear(hidden_dim, 1)
        self.rgb_out = nn.Linear(hidden_dim, 3)

    def forward(self, x, dir_enc=None):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))

        # If direction encoding is given, concatenate it (currently skipped)
        if dir_enc is not None:
            h = torch.cat([h, dir_enc], dim=-1)

        sigma = F.softplus(self.sigma_out(h))  # boost opacity early

        rgb = torch.sigmoid(self.rgb_out(h))  # RGB in [0, 1]

        return torch.cat([rgb, sigma], dim=-1)  # (..., 4)
