import torch
from lvfm.data_generation import burgers_exact_eqn

def phi_xt(x, t):
    return x * (2.0-x) * t

def g_ic(x, t, mu):
    return burgers_exact_eqn(x, torch.zeros_like(t), mu)

def u_constrained(decoder, x, t, alpha, mu):
    return g_ic(x, t, mu) + phi_xt(x, t) * decoder(x, alpha)