import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    '''
    
    '''
    def __init__(self, num_freqs=10, include_input=True, log_sampling=True):
        super().__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., num_freqs - 1, steps=num_freqs)
            # gives [1,2,4,8,...,512]
            #multiplied with input coordinates to modulate sine/cosine functions
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
        # each x becomes [x,sin(1x),cos(1x),sin(2x),cos(2x),…,sin(512x),cos(512x)]

        return torch.cat(out, dim=-1)
        #outputs 21 features per dimension, that makes 21*3 = 63 outputs for 3d i.e, positional dimensions = 63 while constructing NeRFMLP

        #Reason of this transformation can be understood like this:
        '''
        raw x is a straight line
        sin(2x),cos(2x) adds wiggles
        sin(128x),cos(128x) tiny ripples for very fine detail
        '''
    
