import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True, log_sampling=True):
        super().__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., num_freqs - 1, steps=num_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** (num_freqs - 1), steps=num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (..., dim) coordinates (e.g., (N_rays, N_samples, 3))
        Returns:
            encoded: (..., dim * (2 * num_freqs + int(include_input))) positional encoding
        """
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                out.append(func(x * freq))
        return torch.cat(out, dim=-1)
