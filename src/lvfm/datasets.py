import torch
from torch.utils.data import Dataset

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