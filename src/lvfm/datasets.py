import torch
from torch.utils.data import Dataset

class LinearOscillator2DDataset(Dataset):
    def __init__(self, num_batches, num_interior, num_terminal, T, x1_bounds=(-1., 1.), x2_bounds=(-1., 1.), tau_max=None, scale_to_minus1_1=True, scale_time_to_0_1=True):
        super().__init__()
        self.num_batches = num_batches
        self.num_interior = num_interior
        self.num_terminal = num_terminal
        self.T = float(T)
        self.x1_bounds = tuple(x1_bounds)
        self.x2_bounds = tuple(x2_bounds)
        self.tau_max = float(T if tau_max is None else tau_max)
        self.scale_to_minus1_1 = scale_to_minus1_1
        self.scale_time_to_0_1 = scale_time_to_0_1
        self.coordinate_dim = 2

    def __len__(self):
        return self.num_batches

    # sets the max time to sample from
    # picks the maximum time between 0 or the minimum of the full time and tau max
    def set_tau_max(self, tau_max):
        tau_max = float(tau_max)
        self.tau_max = max(0.0, min(self.T, tau_max))

    # sample n numbers uniformly in the range low to high
    def _sample_uniform(self, n, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1)

    # return n samples for each state in the range of their bounds
    def _sample_states_phys(self, n):
        x1 = self._sample_uniform(n, self.x1_bounds)
        x2 = self._sample_uniform(n, self.x2_bounds)
        return torch.cat([x1, x2], dim=-1)

    def _sample_tau_phys(self, n):
        if self.tau_max <= 0:
            return torch.zeros(n, 1)
        return self.tau_max * torch.rand(n, 1)

    # normalize tau to be between 0 and 1
    def _scale_tau(self, tau_phys):
        if not self.scale_time_to_0_1:
            return tau_phys
        return tau_phys / self.T

    def _scale_states(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys
        bounds = torch.tensor([self.x1_bounds, self.x2_bounds], dtype=x_phys.dtype, device=x_phys.device)
        low = bounds[:, 0]
        high = bounds[:, 1]
        return 2.0 * (x_phys - low) / (high-low) - 1.0

    def __getitem__(self, idx):
        # sample interior points
        x_interior_phys = self._sample_states_phys(self.num_interior)
        x_interior_net = self._scale_states(x_interior_phys)
        tau_interior_phys = self._sample_tau_phys(self.num_interior)
        tau_interior_net = self._scale_tau(tau_interior_phys)
        xt_interior = torch.cat([x_interior_net, tau_interior_net], dim=-1)

        # sample terminal points
        x_terminal_phys = self._sample_states_phys(self.num_terminal)
        x_terminal_net = self._scale_states(x_terminal_phys)
        tau_terminal_phys = torch.zeros(self.num_terminal, 1)
        tau_terminal_net = self._scale_tau(tau_terminal_phys)
        xt_terminal = torch.cat([x_terminal_net, tau_terminal_net], dim=-1)

        return {
            "xt_interior": xt_interior.float(),
            "xt_terminal": xt_terminal.float(),
            "x_interior_phys": x_interior_phys.float(),
            "tau_interior_phys": tau_interior_phys.squeeze(-1).float(),
            "x_terminal_phys": x_terminal_phys.float()
        }