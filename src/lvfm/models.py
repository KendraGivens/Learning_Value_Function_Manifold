import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchdiffeq import odeint

class Swish(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

# simple mlp
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, activation_fn=Swish, out_dim=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, in_dim if out_dim is None else out_dim),
        ) 

    def forward(self, x):
        return self.model(x) 

# takes in current time and latent state 
# outputs the time derivative of latent state
class PNODE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, activation_fn=Swish):
        super().__init__()
        self.model = MLP(in_dim=latent_dim+1, hidden_dim=hidden_dim, activation_fn=activation_fn, out_dim=latent_dim)

    def forward(self, t, latent_state):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=latent_state.dtype, device=latent_state.device)
        if t.ndim == 0:
            t_expanded = t.expand(latent_state.shape[:-1] + (1,))
        else:
            t_expanded = t[..., None].expand(latent_state.shape[:-1] + (1,))
        inputs = torch.cat([latent_state, t_expanded], dim=-1)
        return self.model(inputs)


# takes in inputs and latent state alpha
# returns the learned linear projection of the inputs and latent state
class AdditiveConditioning(nn.Module):
    def __init__(self, inputs, latent_state, out):
        super().__init__()
        self.inputs = inputs
        self.latent_state = latent_state
        self.out = out
        self.A = nn.Parameter(torch.empty(self.out, self.latent_state))
        self.B = nn.Parameter(torch.empty(self.out, self.inputs))
        self.bias = nn.Parameter(torch.empty(self.out))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.inputs)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, latent_state):
        latent_state_transform = torch.einsum('...j,oj->...o', latent_state, self.A)
        inputs_transform = torch.einsum('...i,oi->...o', inputs, self.B)
        return latent_state_transform + inputs_transform + self.bias

# takes in spatial coordinates x and latent state alpha 
# multiplies enriched features of x with learned linear projections of previous output and alpha
class MFNBase(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, out_dim, num_layers):
        super().__init__()
        self.conditioning = nn.ModuleList(
            [AdditiveConditioning(in_dim, latent_dim, hidden_dim)] + 
            [AdditiveConditioning(hidden_dim, latent_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.conditioning_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, latent_state):
        out = self.filters[0](x) * self.conditioning[0](x * 0., latent_state)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.conditioning[i](out, latent_state)
        out = self.conditioning_out(out)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

# takes in spatial coordinates x
# applies learned linear projections to them
# passes through sin and cos
# outputs enriched spatial features
class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_scale = weight_scale
        self.reset_parameters()

    # initialize the parameters to kaiming uniform distirbution
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return torch.cat([
            torch.sin(F. linear(x, self.weight * self.weight_scale)),
            torch.cos(F.linear(x, self.weight * self.weight_scale))
        ], dim=-1)


# takes in spatial coordinates x
# applies learned linear projections to them
# passes through sin
# outputs enriched spatial features
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega0=30.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega0 = omega0
        self.is_first = is_first

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega0
            self.weight.uniform_(-bound, bound)
            self.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * F.linear(x, self.weight, self.bias))


# builds the filters for MFN
class FourierNet(MFNBase):
    def __init__(self, in_dim, hidden_dim, latent_dim, out_dim, num_layers=3, input_scale=256.0):
        super().__init__(in_dim, hidden_dim, latent_dim, out_dim, num_layers)
        self.filters = nn.ModuleList(
            [FourierLayer(in_dim, hidden_dim // 2, input_scale / np.sqrt(num_layers+1)) for _ in range(num_layers + 1)]
        )

# builds the filters for MFN
class SirenNet(MFNBase):
    def __init__(self, in_dim, hidden_dim, latent_dim, out_dim, num_layers=3, omega0=30.0):
        super().__init__(in_dim, hidden_dim, latent_dim, out_dim, num_layers)
        self.filters = nn.ModuleList([
            SirenLayer(in_dim, hidden_dim, omega0=omega0, is_first=(i==0)) for i in range(num_layers+1)
        ])

# takes in spatial coordinates x and latents alpha(tau)
# outputs the value function v(x, alpha(tau))
class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, coordinate_dim, out_dim, num_layers, net_type="fourier", input_scale=64, omega0=30.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.coordinate_dim = coordinate_dim
        self.out_dim = out_dim
        if net_type == "fourier":
            self.model = FourierNet(in_dim=self.coordinate_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, out_dim=self.out_dim, num_layers=num_layers, input_scale=input_scale)
        else:
            self.model = SirenNet(in_dim=self.coordinate_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, out_dim=self.out_dim, num_layers=num_layers, omega0=omega0)
    
    def forward(self, x, latents):
        return self.model(x, latents)


class INR_PNODE(nn.Module):
    def __init__(self, latent_dim, decoder, pnode, loss_manager, ode_solver=odeint, method="dopri5", rtol=1e-7, atol=1e-9, device=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = decoder
        self.pnode = pnode
        self.loss_manager = loss_manager
        self.ode_solver = ode_solver
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.device = device

        self.alpha0 = nn.Parameter(torch.zeros(1, self.latent_dim))

    # returns the latents and their time derivatives at each tau qeury
    def compute_latent_and_dlatent(self, tau_queries):
        tau_queries = tau_queries.reshape(-1)
        unique_taus, inverse_indices = torch.unique(tau_queries, sorted=True, return_inverse=True)

        if unique_taus.numel() == 0 or unique_taus[0].item() > 0.0:
            t_eval = torch.cat([torch.zeros(1, device=tau_queries.device, dtype=unique_taus.dtype), unique_taus], dim=0)
            prepend_zero = True
        else:
            t_eval = unique_taus
            prepend_zero = False

        latent_trajectory = self.ode_solver(self.pnode, self.alpha0, t_eval, rtol=self.rtol, atol=self.atol, method=self.method)[:, 0, :]
        dlatent_trajectory = torch.stack([self.pnode(t_eval[i], latent_trajectory[i:i+1]).squeeze(0) for i in range(t_eval.shape[0])], dim=0)

        if prepend_zero:
            unique_latents = latent_trajectory[1:]
            unique_dlatents = dlatent_trajectory[1:]
        else:
            unique_latents = latent_trajectory
            unique_dlatents = dlatent_trajectory

        latents = unique_latents[inverse_indices]
        dlatents = unique_dlatents[inverse_indices]

        return latents, dlatents

    def decode_terminal(self, xt_terminal):
        num_terminal = xt_terminal.shape[0]
        latent0 = self.alpha0.expand(num_terminal, self.latent_dim)
        x_terminal = xt_terminal[:, :self.decoder.coordinate_dim]
        return self.decoder(x_terminal, latent0)

    # computes the latents and dlatents at the time queries
    # passes the spatial coordinates and the latents into the decoder
    # to obtain the value at that coordinate and time 
    def compute_value_at_xt(self, xt, tau_phys):
        latent, dlatent = self.compute_latent_and_dlatent(tau_phys)
        latent = latent.requires_grad_(True)
        x = xt[:, :self.decoder.coordinate_dim]
        
        # compute value at x and latent(tau)
        V = self.decoder(x, latent)

        return V, latent, dlatent
    
    def compute_losses(self, batch, mode="train"):
        xt_interior = batch["xt_interior"].to(self.device).float().requires_grad_(True)
        tau_interior_phys = batch["tau_interior_phys"].to(self.device).float()
        xt_terminal = batch["xt_terminal"].to(self.device).float()
        x_terminal_phys = batch["x_terminal_phys"].to(self.device).float()
    
        V, latent, dlatent = self.compute_value_at_xt(xt_interior, tau_interior_phys)
    
        if mode == "visualization":
            return V.detach().cpu().numpy()
    
        V_terminal = self.decode_terminal(xt_terminal)
    
        loss_dict = self.loss_manager.compute_losses(model=self, xt_interior=xt_interior, x_terminal_phys=x_terminal_phys, V=V, V_terminal=V_terminal, latent=latent, dlatent=dlatent)
    
        loss_train = (self.loss_manager.loss_weights["terminal"] * loss_dict["loss_terminal"] + self.loss_manager.loss_weights["pinn"] * loss_dict["loss_pinn"])
        loss_dict["loss_train"] = loss_train
        return loss_dict

    def create_optimizers(self, lr=1e-4, ode_lr=None):
        if ode_lr is None:
            ode_lr = lr/10

        return {
            "optim_decoder": torch.optim.Adam(self.decoder.parameters(), lr=lr),
            "optim_ode": torch.optim.Adam(self.pnode.parameters(), lr=ode_lr),
            "optim_alpha0": torch.optim.Adam([self.alpha0], lr=lr),
        }