import torch
import torch.nn as nn
import math
from lvfm.helpers import squeeze_last

class HJVIResidualBase(nn.Module):
    def __init__(
        self,
        coordinate_dim,
        radius,
        T=1.0,
        scale_to_minus1_1=True,
        scale_time_to_01=True,
    ):
        super().__init__()
        self.coordinate_dim = int(coordinate_dim)
        self.radius = float(radius)
        self.T = float(T)
        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_01 = bool(scale_time_to_01)

    def target_function(self, x_phys):
        return torch.sqrt(
            x_phys[..., 0] ** 2 + x_phys[..., 1] ** 2 + 1e-12
        ) - self.radius

    def _unscale_time_gradient(self, time_grad):
        if not self.scale_time_to_01:
            return time_grad
        return time_grad / self.T

    def _tau_phys_from_xt(self, xt):
        tau = xt[:, self.coordinate_dim]
        if self.scale_time_to_01:
            tau = tau * self.T
        return tau

    def _spatial_grad(self, V_scalar, xt):
        dV_dxt = torch.autograd.grad(
            V_scalar,
            xt,
            grad_outputs=torch.ones_like(V_scalar),
            retain_graph=True,
            create_graph=True,
        )[0]
        return dV_dxt[:, :self.coordinate_dim], dV_dxt[:, self.coordinate_dim]

    def _hjvi(self, V_scalar, tau_grad_phys, xt, spatial_grad_net):
        x_net = xt[:, :self.coordinate_dim]
        x_phys = self._unscale_x(x_net)
        spatial_grad_phys = self._unscale_spatial_gradient(spatial_grad_net)

        boundary = self.target_function(x_phys)
        hamiltonian = self.compute_hamiltonian(x_phys, spatial_grad_phys)

        return torch.maximum(tau_grad_phys - hamiltonian, V_scalar - boundary)

    def compute_deepreach_residual(self, model, xt):
        xt = xt.requires_grad_(True)

        V = squeeze_last(model(xt))
        spatial_grad_net, tau_grad_net = self._spatial_grad(V, xt)

        tau_grad_phys = self._unscale_time_gradient(tau_grad_net)

        return self._hjvi(
            V_scalar=V,
            tau_grad_phys=tau_grad_phys,
            xt=xt,
            spatial_grad_net=spatial_grad_net,
        )

    def compute_cnf_rom_residual(self, model, V, raw, xt, latent, dlatent):
        V_scalar = squeeze_last(V)
        raw_scalar = squeeze_last(raw)

        spatial_grad_net, _ = self._spatial_grad(V_scalar, xt)

        draw_dalpha = torch.autograd.grad(
            raw_scalar,
            latent,
            grad_outputs=torch.ones_like(raw_scalar),
            retain_graph=True,
            create_graph=True,
        )[0]

        tau_phys = self._tau_phys_from_xt(xt)
        scale = model.value_var / model.value_normto

        tau_grad_phys = scale * (
            raw_scalar + tau_phys * (draw_dalpha * dlatent).sum(dim=-1)
        )

        return self._hjvi(
            V_scalar=V_scalar,
            tau_grad_phys=tau_grad_phys,
            xt=xt,
            spatial_grad_net=spatial_grad_net,
        )

class LinearOscillator2DResidual(HJVIResidualBase):
    def __init__(
        self,
        oscillation_speed=1.0,
        control_bound=1.0,
        disturbance_bound=0.5,
        radius=0.25,
        T=1.0,
        x1_bounds=(-1.0, 1.0),
        x2_bounds=(-1.0, 1.0),
        scale_to_minus1_1=True,
        scale_time_to_01=True,
    ):
        super().__init__(
            coordinate_dim=2,
            radius=radius,
            T=T,
            scale_to_minus1_1=scale_to_minus1_1,
            scale_time_to_01=scale_time_to_01,
        )

        self.oscillation_speed = float(oscillation_speed)
        self.control_bound = float(control_bound)
        self.disturbance_bound = float(disturbance_bound)

        self.x1_bounds = tuple(x1_bounds)
        self.x2_bounds = tuple(x2_bounds)

    def _unscale_x(self, x_net):
        if not self.scale_to_minus1_1:
            return x_net

        x_phys = x_net.clone()

        x_phys[..., 0] = 0.5 * (x_net[..., 0] + 1.0) * (
            self.x1_bounds[1] - self.x1_bounds[0]
        ) + self.x1_bounds[0]

        x_phys[..., 1] = 0.5 * (x_net[..., 1] + 1.0) * (
            self.x2_bounds[1] - self.x2_bounds[0]
        ) + self.x2_bounds[0]

        return x_phys

    def _unscale_spatial_gradient(self, spatial_grad):
        if not self.scale_to_minus1_1:
            return spatial_grad

        grad = spatial_grad.clone()

        grad[..., 0] = spatial_grad[..., 0] * (
            2.0 / (self.x1_bounds[1] - self.x1_bounds[0])
        )

        grad[..., 1] = spatial_grad[..., 1] * (
            2.0 / (self.x2_bounds[1] - self.x2_bounds[0])
        )

        return grad

    def compute_hamiltonian(self, x_phys, spatial_grad):
        x1 = x_phys[..., 0]
        x2 = x_phys[..., 1]

        p1 = spatial_grad[..., 0]
        p2 = spatial_grad[..., 1]

        base = p1 * x2 + p2 * (-(self.oscillation_speed ** 2) * x1)

        control_term = -self.control_bound * torch.abs(p2)
        disturbance_term = self.disturbance_bound * torch.abs(p2)

        return base + control_term + disturbance_term

class Air3DResidual(HJVIResidualBase):
    def __init__(
        self,
        vp=0.75,
        ve=0.75,
        control_bound=3.0,
        disturbance_bound=3.0,
        radius=0.25,
        T=1.0,
        x_bounds=(-2.0, 2.0),
        y_bounds=(-2.0, 2.0),
        psi_bounds=(-math.pi, math.pi),
        scale_to_minus1_1=True,
        scale_time_to_01=True,
        angle_alpha_factor=1.2,
    ):
        super().__init__(
            coordinate_dim=3,
            radius=radius,
            T=T,
            scale_to_minus1_1=scale_to_minus1_1,
            scale_time_to_01=scale_time_to_01,
        )

        self.vp = float(vp)
        self.ve = float(ve)
        self.control_bound = float(control_bound)
        self.disturbance_bound = float(disturbance_bound)

        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.psi_bounds = tuple(psi_bounds)

        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi

    def _unscale_x(self, x_net):
        if not self.scale_to_minus1_1:
            return x_net

        x_phys = x_net.clone()

        x_phys[..., 0] = 0.5 * (x_net[..., 0] + 1.0) * (
            self.x_bounds[1] - self.x_bounds[0]
        ) + self.x_bounds[0]

        x_phys[..., 1] = 0.5 * (x_net[..., 1] + 1.0) * (
            self.y_bounds[1] - self.y_bounds[0]
        ) + self.y_bounds[0]

        # DeepReach-style angle scaling.
        x_phys[..., 2] = x_net[..., 2] * self.angle_scale

        return x_phys

    def _unscale_spatial_gradient(self, spatial_grad):
        if not self.scale_to_minus1_1:
            return spatial_grad

        grad = spatial_grad.clone()

        grad[..., 0] = spatial_grad[..., 0] * (
            2.0 / (self.x_bounds[1] - self.x_bounds[0])
        )

        grad[..., 1] = spatial_grad[..., 1] * (
            2.0 / (self.y_bounds[1] - self.y_bounds[0])
        )

        grad[..., 2] = spatial_grad[..., 2] / self.angle_scale

        return grad

    def compute_hamiltonian(self, x_phys, spatial_grad):
        x = x_phys[..., 0]
        y = x_phys[..., 1]
        psi = x_phys[..., 2]

        px = spatial_grad[..., 0]
        py = spatial_grad[..., 1]
        ppsi = spatial_grad[..., 2]

        base = (
            px * (-self.ve + self.vp * torch.cos(psi))
            + py * (self.vp * torch.sin(psi))
        )

        control_coeff = px * y - py * x - ppsi
        control_term = self.control_bound * torch.abs(control_coeff)

        disturbance_term = -self.disturbance_bound * torch.abs(ppsi)

        return base + control_term + disturbance_term