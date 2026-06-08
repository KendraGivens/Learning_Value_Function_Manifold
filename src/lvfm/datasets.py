import torch
import math
from torch.utils.data import Dataset

class LinearOscillator2DDataset(Dataset):
    def __init__(
        self,
        num_batches,
        num_interior,
        num_terminal,
        T,
        x1_bounds=(-1.0, 1.0),
        x2_bounds=(-1.0, 1.0),
        tau_max=None,
        scale_to_minus1_1=True,
        scale_time_to_0_1=True,
        efficient=False,
        num_unique_taus=8,
    ):
        super().__init__()

        self.num_batches = int(num_batches)
        self.num_interior = int(num_interior)
        self.num_terminal = int(num_terminal)
        self.T = float(T)

        self.x1_bounds = tuple(x1_bounds)
        self.x2_bounds = tuple(x2_bounds)

        self.tau_max = float(self.T if tau_max is None else tau_max)

        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_0_1 = bool(scale_time_to_0_1)

        self.efficient = bool(efficient)
        self.num_unique_taus = num_unique_taus

        self.coordinate_dim = 2

    def __len__(self):
        return self.num_batches

    def set_tau_max(self, tau_max):
        tau_max = float(tau_max)
        self.tau_max = max(0.0, min(self.T, tau_max))

    def _sample_uniform(self, n, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1)

    def _sample_states_phys(self, n):
        x1 = self._sample_uniform(n, self.x1_bounds)
        x2 = self._sample_uniform(n, self.x2_bounds)
        return torch.cat([x1, x2], dim=-1)

    def _sample_tau_phys(self, n):
        if self.tau_max <= 0.0:
            return torch.zeros(n, 1)

        # Non-efficient: independent random tau for every point.
        if not self.efficient:
            return self.tau_max * torch.rand(n, 1)

        # Efficient: reuse only num_unique_taus different tau values.
        if self.num_unique_taus is None or self.num_unique_taus >= n:
            return self.tau_max * torch.rand(n, 1)

        k = max(1, int(self.num_unique_taus))

        edges = torch.linspace(0.0, self.tau_max, steps=k + 1)

        tau_unique = []
        for i in range(k):
            low = edges[i]
            high = edges[i + 1]
            tau_unique.append(low + (high - low) * torch.rand(1))

        tau_unique = torch.stack(tau_unique).reshape(k, 1)

        idx = torch.randint(0, k, (n,))
        return tau_unique[idx]

    def _scale_tau(self, tau_phys):
        if not self.scale_time_to_0_1:
            return tau_phys
        return tau_phys / self.T

    def _scale_states(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys

        bounds = torch.tensor(
            [self.x1_bounds, self.x2_bounds],
            dtype=x_phys.dtype,
            device=x_phys.device,
        )

        low = bounds[:, 0]
        high = bounds[:, 1]

        return 2.0 * (x_phys - low) / (high - low) - 1.0

    def __getitem__(self, idx):
        x_interior_phys = self._sample_states_phys(self.num_interior)
        x_interior_net = self._scale_states(x_interior_phys)

        tau_interior_phys = self._sample_tau_phys(self.num_interior)
        tau_interior_net = self._scale_tau(tau_interior_phys)

        xt_interior = torch.cat([x_interior_net, tau_interior_net], dim=-1)

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
            "x_terminal_phys": x_terminal_phys.float(),
        }


class Air3DDataset(Dataset):
    def __init__(
        self,
        num_batches,
        num_interior,
        num_terminal,
        T,
        x_bounds=(-2.0, 2.0),
        y_bounds=(-2.0, 2.0),
        psi_bounds=(-math.pi, math.pi),
        tau_max=None,
        scale_to_minus1_1=True,
        scale_time_to_0_1=True,
        efficient=False,
        num_unique_taus=8,
        angle_alpha_factor=1.2
    ):
        super().__init__()

        self.num_batches = int(num_batches)
        self.num_interior = int(num_interior)
        self.num_terminal = int(num_terminal)
        self.T = float(T)

        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.psi_bounds = tuple(psi_bounds)

        self.tau_max = float(self.T if tau_max is None else tau_max)

        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_0_1 = bool(scale_time_to_0_1)

        self.efficient = bool(efficient)
        self.num_unique_taus = num_unique_taus

        self.coordinate_dim = 3

        # DeepReach-style angle scaling:
        # psi_net = psi_phys / (angle_alpha_factor * pi)
        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi

    def __len__(self):
        return self.num_batches

    def set_tau_max(self, tau_max):
        tau_max = float(tau_max)
        self.tau_max = max(0.0, min(self.T, tau_max))

    def _sample_uniform(self, n, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1)

    def _sample_states_phys(self, n):
        # Always uniform now. No boundary-biased sampling.
        x = self._sample_uniform(n, self.x_bounds)
        y = self._sample_uniform(n, self.y_bounds)
        psi = self._sample_uniform(n, self.psi_bounds)
        return torch.cat([x, y, psi], dim=-1)

    def _sample_tau_phys(self, n):
        if self.tau_max <= 0.0:
            return torch.zeros(n, 1)

        # Non-efficient: independent random tau for every point.
        if not self.efficient:
            return self.tau_max * torch.rand(n, 1)

        # Efficient: reuse only num_unique_taus different tau values.
        if self.num_unique_taus is None or self.num_unique_taus >= n:
            return self.tau_max * torch.rand(n, 1)

        k = max(1, int(self.num_unique_taus))

        edges = torch.linspace(0.0, self.tau_max, steps=k + 1)

        tau_unique = []
        for i in range(k):
            low = edges[i]
            high = edges[i + 1]
            tau_unique.append(low + (high - low) * torch.rand(1))

        tau_unique = torch.stack(tau_unique).reshape(k, 1)

        idx = torch.randint(0, k, (n,))
        return tau_unique[idx]

    def _scale_tau(self, tau_phys):
        if not self.scale_time_to_0_1:
            return tau_phys
        return tau_phys / self.T

    def _scale_states(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys

        x_net = x_phys.clone()

        x_net[:, 0] = 2.0 * (x_phys[:, 0] - self.x_bounds[0]) / (
            self.x_bounds[1] - self.x_bounds[0]
        ) - 1.0

        x_net[:, 1] = 2.0 * (x_phys[:, 1] - self.y_bounds[0]) / (
            self.y_bounds[1] - self.y_bounds[0]
        ) - 1.0

        # Keep DeepReach-style angle scaling.
        x_net[:, 2] = x_phys[:, 2] / self.angle_scale

        return x_net

    def __getitem__(self, idx):
        x_interior_phys = self._sample_states_phys(self.num_interior)
        x_interior_net = self._scale_states(x_interior_phys)

        tau_interior_phys = self._sample_tau_phys(self.num_interior)
        tau_interior_net = self._scale_tau(tau_interior_phys)

        xt_interior = torch.cat([x_interior_net, tau_interior_net], dim=-1)

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
            "x_terminal_phys": x_terminal_phys.float(),
        }

class Air6DJointDataset(Dataset):
    def __init__(
        self,
        num_batches,
        num_interior,
        num_terminal,
        T,
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        theta_bounds=(-math.pi, math.pi),
        tau_max=None,
        scale_to_minus1_1=True,
        scale_time_to_0_1=True,
        efficient=False,
        num_unique_taus=16,
        angle_alpha_factor=1.2,
    ):
        super().__init__()

        self.num_batches = int(num_batches)
        self.num_interior = int(num_interior)
        self.num_terminal = int(num_terminal)
        self.T = float(T)

        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.theta_bounds = tuple(theta_bounds)

        self.tau_max = float(self.T if tau_max is None else tau_max)

        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_0_1 = bool(scale_time_to_0_1)

        self.efficient = bool(efficient)
        self.num_unique_taus = num_unique_taus

        self.coordinate_dim = 6

        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi

    def __len__(self):
        return self.num_batches

    def set_tau_max(self, tau_max):
        tau_max = float(tau_max)
        self.tau_max = max(0.0, min(self.T, tau_max))

    def _sample_uniform(self, n, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1)

    def _sample_states_phys(self, n):
        xp = self._sample_uniform(n, self.x_bounds)
        yp = self._sample_uniform(n, self.y_bounds)
        thp = self._sample_uniform(n, self.theta_bounds)

        xe = self._sample_uniform(n, self.x_bounds)
        ye = self._sample_uniform(n, self.y_bounds)
        the = self._sample_uniform(n, self.theta_bounds)

        return torch.cat([xp, yp, thp, xe, ye, the], dim=-1)

    def _sample_tau_phys(self, n):
        if self.tau_max <= 0.0:
            return torch.zeros(n, 1)

        if not self.efficient:
            return self.tau_max * torch.rand(n, 1)

        if self.num_unique_taus is None or self.num_unique_taus >= n:
            return self.tau_max * torch.rand(n, 1)

        k = max(1, int(self.num_unique_taus))
        edges = torch.linspace(0.0, self.tau_max, steps=k + 1)

        tau_unique = []
        for i in range(k):
            low = edges[i]
            high = edges[i + 1]
            tau_unique.append(low + (high - low) * torch.rand(1))

        tau_unique = torch.stack(tau_unique).reshape(k, 1)
        idx = torch.randint(0, k, (n,))

        return tau_unique[idx]

    def _scale_tau(self, tau_phys):
        if not self.scale_time_to_0_1:
            return tau_phys
        return tau_phys / self.T

    def _scale_states(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys

        x_net = x_phys.clone()

        # pursuer position
        x_net[:, 0] = 2.0 * (x_phys[:, 0] - self.x_bounds[0]) / (
            self.x_bounds[1] - self.x_bounds[0]
        ) - 1.0

        x_net[:, 1] = 2.0 * (x_phys[:, 1] - self.y_bounds[0]) / (
            self.y_bounds[1] - self.y_bounds[0]
        ) - 1.0

        # pursuer heading
        x_net[:, 2] = x_phys[:, 2] / self.angle_scale

        # evader position
        x_net[:, 3] = 2.0 * (x_phys[:, 3] - self.x_bounds[0]) / (
            self.x_bounds[1] - self.x_bounds[0]
        ) - 1.0

        x_net[:, 4] = 2.0 * (x_phys[:, 4] - self.y_bounds[0]) / (
            self.y_bounds[1] - self.y_bounds[0]
        ) - 1.0

        # evader heading
        x_net[:, 5] = x_phys[:, 5] / self.angle_scale

        return x_net

    def __getitem__(self, idx):
        x_interior_phys = self._sample_states_phys(self.num_interior)
        x_interior_net = self._scale_states(x_interior_phys)

        tau_interior_phys = self._sample_tau_phys(self.num_interior)
        tau_interior_net = self._scale_tau(tau_interior_phys)

        xt_interior = torch.cat([x_interior_net, tau_interior_net], dim=-1)

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
            "x_terminal_phys": x_terminal_phys.float(),
        }