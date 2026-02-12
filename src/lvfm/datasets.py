import torch
from torch.utils.data import Dataset

class Air3DPhysicsDataset(Dataset):
    def __init__(self, n_samples, T=1.0, tau_eps=1e-4, tau_max=None, tau_power=1.0, x1_range=(-1.0, 1.0), x2_range=(-1.0, 1.0), x3_range=(-torch.pi, torch.pi)):
        super().__init__()
        self.n_samples = n_samples
        self.T = float(T)
        self.tau_eps = float(tau_eps)

        self.tau_max = float(tau_max) if tau_max is not None else self.T
        self.tau_power = float(tau_power)

        self.x1_range = x1_range
        self.x2_range = x2_range
        self.x3_range = x3_range

    def set_tau_schedule(self, tau_max, tau_power=None):
        self.tau_max = float(min(max(tau_max, self.tau_eps), self.T))
        if tau_power is not None:
            self.tau_power = float(tau_power)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x1 = torch.rand(()) * (self.x1_range[1] - self.x1_range[0]) + self.x1_range[0]
        x2 = torch.rand(()) * (self.x2_range[1] - self.x2_range[0]) + self.x2_range[0]
        x3 = torch.rand(()) * (self.x3_range[1] - self.x3_range[0]) + self.x3_range[0]
        x = torch.stack([x1, x2, x3], dim=0)  # (3,)

        u = torch.rand(())
        tau_hi = max(self.tau_max, self.tau_eps)
        tau = self.tau_eps + (tau_hi - self.tau_eps) * (u ** self.tau_power)

        return x, tau

# samples (x, t, mu) with target u from the solution grid
class BurgersExactDataset(Dataset):
    def __init__(self, x_axis, t_axis, mu_axis, u_grid, n_samples=200000):
        super().__init__()
        # coordinate axes
        self.x_axis = x_axis
        self.t_axis = t_axis
        self.mu_axis = mu_axis

        # solution field
        self.u_grid = u_grid

        # grid sizes
        self.n_mu, self.n_tp1, self.n_x = u_grid.shape

        self.n_samples = int(n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # sample random axis indices
        i_mu = torch.randint(0, self.n_mu, (1,)).item()
        i_t = torch.randint(1, self.n_tp1, (1,)).item()
        i_x = torch.randint(0, self.n_x, (1,)).item()

        # coordinates at those indices
        x_coord = self.x_axis[i_x]
        t_coord = self.t_axis[i_t]
        mu_val = self.mu_axis[i_mu]

        # target solution value
        y = self.u_grid[i_mu, i_t, i_x]

        return x_coord, t_coord, mu_val, y

class BurgersPhysicsDataset(Dataset):
    def __init__(self, mus, n_samples):
        super().__init__()
        self.mus = torch.tensor(mus)
        self.n_mu = len(self.mus)
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        i_mu = torch.randint(0, self.n_mu, (1,)).item()
        mu_val = self.mus[i_mu]

        eps = 1e-6
        x_coord = torch.clamp(torch.rand(())*2.0, eps, 2.0-eps)
        t_coord = torch.clamp(torch.rand(()), eps, 1.0)

        return x_coord, t_coord, mu_val