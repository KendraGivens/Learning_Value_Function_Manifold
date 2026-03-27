import torch
import torch.nn as nn

class LinearOscillator2DResidual(nn.Module):
    def __init__(self, oscillation_speed=1.0, control_bound=1.0, disturbance_bound=0.5, radius=0.25, T=1.0, x1_bounds=(-1.0, 1.0), x2_bounds=(-1.0, 1.0), scale_to_minus1_1=True, scale_time_to_01=True):
        super().__init__()
        self.oscillation_speed = float(oscillation_speed)
        self.control_bound = float(control_bound)
        self.disturbance_bound = float(disturbance_bound)
        self.radius = float(radius)
        self.T = float(T)
        self.x1_bounds = tuple(x1_bounds)
        self.x2_bounds = tuple(x2_bounds)
        self.scale_to_minus1_1 = scale_to_minus1_1
        self.scale_time_to_01 = scale_time_to_01

    # defines the target set as a circle 
    def target_function(self, x_phys):
        return torch.sqrt(x_phys[..., 0]**2 + x_phys[..., 1]**2 + 1e-12) - self.radius

    # unscale the coordinates
    def _unscale_x(self, x_net):
        if not self.scale_to_minus1_1:
            return x_net
        bounds = torch.tensor([self.x1_bounds, self.x2_bounds], dtype=x_net.dtype, device=x_net.device)
        low = bounds[:, 0]
        high = bounds[:, 1]
        return 0.5 * (x_net + 1.0) * (high - low) + low

    # unscales the spatial gradient of the network
    def _unscale_spatial_gradient(self, spatial_grad):
        if not self.scale_to_minus1_1:
            return spatial_grad
        bounds = torch.tensor([self.x1_bounds, self.x2_bounds], dtype=spatial_grad.dtype, device=spatial_grad.device)
        low = bounds[:, 0]
        high = bounds[:, 1]
        return spatial_grad * (2.0 / (high - low))

    # unscales the time derivative of the 
    def _unscale_time_gradient(self, time_grad):
        if not self.scale_time_to_01:
            return time_grad
        return time_grad / self.T

    # computes the hamiltonian 
    def compute_hamiltonian(self, x_phys, spatial_grad):
        x1 = x_phys[..., 0]
        x2 = x_phys[..., 1]
        partial_x1 = spatial_grad[..., 0]
        partial_x2 = spatial_grad[..., 1]

        base = partial_x1 * x2 + partial_x2 * (-(self.oscillation_speed**2) * x1)
        control_term = -self.control_bound * torch.abs(partial_x2)
        disturbance_term = self.disturbance_bound * torch.abs(partial_x2)
        
        return base + control_term + disturbance_term

    def compute_residual(self, model, V, xt, latent, dlatent):
        dV_dxt, dV_dalpha = torch.autograd.grad(V, [xt, latent], grad_outputs=torch.ones_like(V), retain_graph=True, create_graph=True)

        spatial_grad_net = dV_dxt[:, :2]
        tau_grad_net = (dV_dalpha * dlatent).sum(dim=-1)

        x_net = xt[:, :2]
        x_phys = self._unscale_x(x_net)
        spatial_grad_phys = self._unscale_spatial_gradient(spatial_grad_net)
        tau_grad_phys = self._unscale_time_gradient(tau_grad_net)

        level_set_x = self.target_function(x_phys)
        hamiltonian = self.compute_hamiltonian(x_phys, spatial_grad_phys)

        time_grad_phys = - tau_grad_phys
        V_scalar = V.squeeze(-1) if V.ndim > 1 else V

        residual = torch.minimum(time_grad_phys + hamiltonian, level_set_x - V_scalar)
        return residual

    def compute_loss(self, model, V, xt, latent, dlatent):
        residual = self.compute_residual(model, V, xt, latent, dlatent)
        loss_pinn = torch.mean(torch.abs(residual))
        return loss_pinn, residual