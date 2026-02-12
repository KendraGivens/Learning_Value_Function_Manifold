import torch
from lvfm.data_generation import burgers_exact_eqn

# analytical computation of the Hamiltonian
def compute_hamiltonian(x, gradV, wmax=3.0, ve=0.75, vp=0.75):
    x1, x2, x3 = x.T
    p1, p2, p3 = gradV.T

    base = p1 * (-ve + vp * torch.cos(x3)) + p2 * (vp * torch.sin(x3))
    a = p1 * x2 - p2 * x1 - p3
    H = base - wmax * torch.abs(a) + wmax * p3
    return H

# computes the signed distance to the target set
# target set is less than the radius
def compute_signed_distance(x, radius):
    l2 = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
    return l2 - radius

# ramp up tau
def tau_curriculum(epoch, max_epochs, T, frac_ramp=0.4, tau0_frac=0.1, power0=3.0):
    ramp_epochs = max(1, int(frac_ramp * max_epochs))
    tau0 = tau0_frac * T
    if epoch >= ramp_epochs:
        tau_max = T
        tau_power = 1.0
    else:
        s = epoch / ramp_epochs
        tau_max = tau0 + (T - tau0) * s
        tau_power = power0 - (power0 - 1.0) * s
    return tau_max, tau_power

# satisfy the initial/terminal condition
def v_constrained(decoder, x, tau, alpha, radius=0.25):
    distance = compute_signed_distance(x, radius)
    D = decoder(x, alpha)
    return distance + tau * D

def phi_xt(x, t):
    return x * (2.0-x) * t

def g_ic(x, t, mu):
    return burgers_exact_eqn(x, torch.zeros_like(t), mu)

def u_constrained(decoder, x, t, alpha, mu):
    return g_ic(x, t, mu) + phi_xt(x, t) * decoder(x, alpha)