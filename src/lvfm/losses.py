import torch
from lvfm.functions import phi_xt, g_ic, u_constrained, compute_signed_distance, compute_hamiltonian

# enforce hji variational inequality         
def Air3Dphysics_loss_fn(model, batch, radius=0.25, ve=0.75, vp=0.75, wmax=3.0):
    x, tau = batch
    x = x.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)

    alpha, f_theta = model.get_latents(tau)
    
    D = model.decoder(x, alpha)
    distance = compute_signed_distance(x, radius)
    V = distance + tau * D

    gradV = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    grad_alpha_D = torch.autograd.grad(D.sum(), alpha, create_graph=True)[0]
    D_tau = (grad_alpha_D * f_theta).sum(dim=1)
    V_tau = D + tau * D_tau

    H = compute_hamiltonian(x, gradV, ve=ve, vp=vp, wmax=wmax)
    residual = torch.minimum(-V_tau + H, distance - V)

    return torch.mean(residual**2)

def data_loss_fn(model, batch):
    x, t, mu, y = batch
    pred = model(x, t, mu)
    return torch.mean((pred-y)**2)

def physics_loss_fn(model, batch):
    x, t, mu = batch
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    mu = mu.clone().detach()

    alpha, f_theta = model.get_latents(t, mu)
    D = model.decoder(x, alpha)
    phi = phi_xt(x, t)
    g = g_ic(x, t, mu)

    dt_phi = torch.autograd.grad(phi.sum(), t, create_graph=True)[0]
    dt_g = torch.zeros_like(t)

    u_t_explicit = dt_g + D * dt_phi
    grad_alpha_D = torch.autograd.grad(D.sum(), alpha, create_graph=True)[0]
    chain_rule_term = (grad_alpha_D * f_theta).sum(dim=1)
     
    u_t_implicit = phi * chain_rule_term
    u_t = u_t_explicit + u_t_implicit
    u_full = g + phi * D
    u_x = torch.autograd.grad(u_full.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    residual = u_t + u_full * u_x - (1.0/mu) * u_xx

    return torch.mean(residual**2)