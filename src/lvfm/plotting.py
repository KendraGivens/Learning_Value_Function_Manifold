import torch
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lvfm.data_generation import burgers_exact_eqn

def save_loss_plot(model, plot_dir):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    data_loss = np.asarray(model.epoch_loss_hist.get("data", []), dtype=float)
    phys_loss = np.asarray(model.epoch_loss_hist.get("physics", []), dtype=float)
    if data_loss.size + phys_loss.size == 0:
        return

    fig = plt.figure(figsize=(8, 4))
    if data_loss.size:
        plt.plot(data_loss, label="Data Loss")
    if phys_loss.size:
        x0 = len(data_loss)
        plt.plot(np.arange(x0, x0 + len(phys_loss)), phys_loss, label="Physics Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epoch")
    fig.savefig(plot_dir / "loss_vs_epoch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

@torch.no_grad()
def plot_heatmap_compare(cnf_pre, cnf_fin, plot_dir, mu0=20.0, n_x=128, n_t=128, T=1.0):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = next(cnf_fin.parameters()).device
    x = torch.linspace(0.0, 2.0, n_x, device=device)
    t = torch.linspace(0.0, T, n_t, device=device)
    X, Tt = torch.meshgrid(x, t, indexing="xy")

    x_flat = X.reshape(-1)
    t_flat = Tt.reshape(-1)
    mu_flat = torch.full_like(x_flat, float(mu0))

    u_true = burgers_exact_eqn(X, Tt, torch.full_like(X, float(mu0)))
    u_pre = cnf_pre(x_flat, t_flat, mu_flat).reshape_as(X)
    u_fin = cnf_fin(x_flat, t_flat, mu_flat).reshape_as(X)

    err_pre = (u_pre - u_true).abs()
    err_fin = (u_fin - u_true).abs()

    def rel_L2(u_pred, u_true):
        return (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-12)).item()

    rel_pre = rel_L2(u_pre, u_true)
    rel_fin = rel_L2(u_fin, u_true)

    u_min = torch.min(torch.stack([u_true.min(), u_pre.min(), u_fin.min()])).item()
    u_max = torch.max(torch.stack([u_true.max(), u_pre.max(), u_fin.max()])).item()
    e_max = torch.max(torch.stack([err_pre.max(), err_fin.max()])).item()

    def show(ax, M, title, vmin=None, vmax=None):
        im = ax.imshow(M.T.detach().cpu(), origin="lower", aspect="auto", extent=[x.min().item(), x.max().item(), t.min().item(), t.max().item()], vmin=vmin, vmax=vmax)
        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(title)
        return im

    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    im0 = show(axs[0,0], u_true, "True u(x,t)", vmin=u_min, vmax=u_max)
    show(axs[0,1], u_pre, f"Pretrained\nRel L2={rel_pre:.3f}", vmin=u_min, vmax=u_max)
    show(axs[0,2], u_fin, f"Finetuned\nRel L2={rel_fin:.3f}", vmin=u_min, vmax=u_max)

    axs[1,0].axis("off")
    im4 = show(axs[1,1], err_pre, "Pretrained Error", vmin=0.0, vmax=e_max)
    show(axs[1,2], err_fin, "Finetuned Error", vmin=0.0, vmax=e_max)

    fig.colorbar(im0, ax=axs[0,:], fraction=0.02)
    fig.colorbar(im4, ax=axs[1,1:], fraction=0.02)
    plt.tight_layout()
    fig.savefig(plot_dir / f"heatmap_compare_mu_{float(mu0):.1f}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

@torch.no_grad()
def plot_heatmap_physics(cnf, plot_dir, mu0=20.0, n_x=128, n_t=128, T=1.0):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = next(cnf.parameters()).device
    x = torch.linspace(0.0, 2.0, n_x, device=device)
    t = torch.linspace(0.0, T, n_t, device=device)
    X, Tt = torch.meshgrid(x, t, indexing="xy")

    x_flat = X.reshape(-1)
    t_flat = Tt.reshape(-1)
    mu_flat = torch.full_like(x_flat, float(mu0))

    u_true = burgers_exact_eqn(X, Tt, torch.full_like(X, float(mu0)))
    u_pred = cnf(x_flat, t_flat, mu_flat).reshape_as(X)
    err = (u_pred - u_true).abs()

    rel_l2 = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-12)).item()

    u_min = torch.min(torch.stack([u_true.min(), u_pred.min()])).item()
    u_max = torch.max(torch.stack([u_true.max(), u_pred.max()])).item()
    e_max = err.max().item()

    def show(ax, M, title, vmin=None, vmax=None):
        im = ax.imshow(M.T.detach().cpu(), origin="lower", aspect="auto",
                       extent=[x.min().item(), x.max().item(), t.min().item(), t.max().item()],
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(title)
        return im

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    im0 = show(axs[0], u_true, "True u(x,t)", vmin=u_min, vmax=u_max)
    im1 = show(axs[1], u_pred, f"Physics Prediction\nRel L2={rel_l2:.3f}", vmin=u_min, vmax=u_max)
    im2 = show(axs[2], err, "Abs Error", vmin=0.0, vmax=e_max)

    fig.colorbar(im0, ax=axs[0], fraction=0.04)
    fig.colorbar(im2, ax=axs[2], fraction=0.04)
    plt.tight_layout()

    fig.savefig(plot_dir / f"physics_result_mu_{float(mu0):.1f}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
