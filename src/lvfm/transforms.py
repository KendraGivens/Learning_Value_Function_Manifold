import torch
import torch.nn as nn

# pass inputs through fourier features
class FourierFeatureTransform(nn.Module):
    def __init__(self, n_freqs=16, max_freq=10.0):
        super().__init__()
        freqs = torch.linspace(1.0, max_freq, n_freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, x): 
        if x.dim() == 1:
            x = x[:, None]
        w = x * self.freqs[None, :] * torch.pi # (B, n_freq)
        return torch.cat([torch.sin(w), torch.cos(w)], dim=-1) # (B, 2*n_freq)