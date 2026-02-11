import torch
import torch.nn as nn
import lightning as L
from torchdiffeq import odeint
from lvfm.losses import data_loss_fn, physics_loss_fn
from lvfm.transforms import FourierFeatureTransform
from lvfm.functions import u_constrained

# decoder that takes in x and alpha and outputs u
class Decoder(nn.Module):
    def __init__(self, latent_dim=10, n_freqs=16, max_freq=10.0, hidden=128, n_hidden_layers=3):
        super().__init__()
        self.ff = FourierFeatureTransform(n_freqs=n_freqs, max_freq=max_freq)
        dim = 2 * n_freqs + latent_dim
        
        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(dim, hidden), nn.Tanh()]
            dim = hidden
        layers += [nn.Linear(dim, 1)]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, alpha):
        phi_x = self.ff(x) # (B, 2*n_freq)
        inputs = torch.cat([phi_x, alpha], dim=-1)
        u = self.model(inputs) # (B, 1)
        return u.squeeze(-1)

# parameterized neural ode that takes in mu, alpha and t and outputs time derivative of alpha
class PNODEFunc(nn.Module):
    def __init__(self, latent_dim=10, hidden=128, n_hidden_layers=2):
        super().__init__()
        dim = latent_dim + 2 # alpha + t + mu

        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(dim, hidden), nn.Tanh()]
            dim = hidden
        layers += [nn.Linear(dim, latent_dim)]

        self.model = nn.Sequential(*layers)

    def forward(self, alpha, t, mu):
        B = alpha.shape[0]
        t_col = t.expand(B, 1)
        mu_col = mu.view(B, 1)
        inputs = torch.cat([alpha, t_col, mu_col], dim=-1)
        return self.model(inputs)

class PNODE(nn.Module):
    def __init__(self, func: PNODEFunc, latent_dim=10):
        super().__init__()
        self.func = func
        self.latent_dim = latent_dim

    def forward(self, t, alpha, mu):
        B = alpha.shape[0]
        t_vec = t.expand(B, 1)
        if mu.dim() == 1:
            mu = mu.unsqueeze(-1)
        return self.func(alpha, t_vec, mu) 

class CNFROM(nn.Module):
    def __init__(self, decoder: Decoder, pnode: PNODE):
        super().__init__()
        self.decoder = decoder
        self.pnode = pnode

    def get_latents(self, t, mu, T=1.0, steps=101):
        device = mu.device
        
        t_grid = torch.linspace(0.0, T, steps, device=device)
        alpha0 = torch.zeros(mu.shape[0], self.pnode.latent_dim, device=device)

        def func(t_, a_):
            return self.pnode(t_, a_, mu)
        alpha_trajectory = odeint(func, alpha0, t_grid, method="rk4")

        # interpolate to t
        dt = t_grid[1] - t_grid[0]
        indices = torch.floor(t/dt).long().clamp(0, steps-2)
        trajectory = alpha_trajectory.permute(1, 0, 2)
        batch_ids = torch.arange(t.shape[0], device=device)
        alpha_start = trajectory[batch_ids, indices]
        alpha_end = trajectory[batch_ids, indices+1]

        t_start = t_grid[indices]
        ratio = ((t-t_start)/dt).unsqueeze(-1)
        alpha_interp = alpha_start + ratio * (alpha_end - alpha_start)

        f_theta = self.pnode(t.unsqueeze(-1), alpha_interp, mu)

        return alpha_interp, f_theta

    def forward(self, x, t, mu):
        alpha, _ = self.get_latents(t, mu)   
        u = u_constrained(self.decoder, x, t, alpha, mu) 
        return u

class ModelWrapper(L.LightningModule):
    def __init__(self, model, lr=1e-3, mode="data"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.mode = mode

        self.loss_hist = {"data": [], "physics": []}
        self.epoch_loss_hist = {"data": [], "physics": []}
        self.epoch_losses = []
    
    def training_step(self, batch, batch_idx):
        if self.mode == "data":
            loss = data_loss_fn(self.model, batch) 
        else:
            loss = physics_loss_fn(self.model, batch)
            
        self.log(f"train_loss_{self.mode}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.epoch_losses.append(loss.detach())
        
        return loss

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.optim.Adam(params, lr=self.lr)


    def on_train_epoch_end(self):
        if len(self.epoch_losses) > 0:
            mean_loss = torch.stack(self.epoch_losses).mean().item()
            self.epoch_loss_hist[self.mode].append(mean_loss)
        self.epoch_losses = []