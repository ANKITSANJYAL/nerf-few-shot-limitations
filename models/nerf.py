import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, num_freqs=10, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.in_dim = in_dim

    def forward(self, x):
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            for fn in [torch.sin, torch.cos]:
                out.append(fn(x * freq))
        return torch.cat(out, dim=-1)  # (B, in_dim * (2*num_freqs + 1))

class TinyNeRF(nn.Module):
    def __init__(self, 
                 pos_in_dim=3, 
                 dir_in_dim=3,
                 hidden_dim=256, 
                 num_layers=8,
                 pos_enc_freqs=10,
                 dir_enc_freqs=4,
                 condition_dim=0):  # e.g., DINO CLS token or FiLM params
        super().__init__()

        self.pos_encoder = PositionalEncoding(pos_in_dim, pos_enc_freqs)
        self.dir_encoder = PositionalEncoding(dir_in_dim, dir_enc_freqs)

        pe_dim = pos_in_dim * (2 * pos_enc_freqs + 1)
        de_dim = dir_in_dim * (2 * dir_enc_freqs + 1)

        input_dim = pe_dim + condition_dim

        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

        self.density_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim + de_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, x, d, condition=None):
        """
        x: (B, 3) 3D point coordinates
        d: (B, 3) viewing directions
        condition: (B, C) optional conditioning vector (e.g., DINOv2 token)
        """
        x_encoded = self.pos_encoder(x)
        if condition is not None:
            x_encoded = torch.cat([x_encoded, condition], dim=-1)

        h = self.mlp(x_encoded)
        sigma = self.density_head(h)

        d_encoded = self.dir_encoder(d)
        color = self.color_head(torch.cat([h, d_encoded], dim=-1))

        return color, sigma
