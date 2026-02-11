import torch

# maps x,t,mu to u (pde solution)
def burgers_exact_eqn(x, t, mu):
    pi = torch.pi
    e1 = torch.exp(-(pi**2)*t/mu)
    e4 = torch.exp(-(4*pi**2)*t/mu)
    
    num = 0.25 * e1 * torch.sin(pi*x) + e4 * torch.sin(2*pi*x)
    den = 1.0 + 0.25 * e1 * torch.cos(pi*x) + 0.5 * e4 * torch.cos(2*pi*x)

    return (2*pi/mu)*(num/den)

def generate_burgers_solution_grid(mu_values, n_x, n_t, T_final=1.0):
    # create range of values as vectors
    x_axis = torch.linspace(0.0, 2.0, n_x)          # (nx,)
    t_axis = torch.linspace(0.0, T_final, n_t + 1)  # (nt+1,)
    mu_axis = torch.tensor(mu_values)                # (nmu,)

    # create full grid across all space, time, parameters
    X_grid = x_axis[None, None, :].expand(mu_axis.shape[0], t_axis.shape[0], x_axis.shape[0])
    T_grid = t_axis[None, :, None].expand_as(X_grid)
    Mu_grid = mu_axis[:, None, None].expand_as(X_grid)

    # evaluate solution on full grid
    u_grid = burgers_exact_eqn(X_grid, T_grid, Mu_grid)

    # enforce boundary conditions
    u_grid[:, :, 0]  = 0.0
    u_grid[:, :, -1] = 0.0

    return x_axis, t_axis, mu_axis, u_grid