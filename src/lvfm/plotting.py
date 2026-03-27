import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path 

@torch.no_grad()
def plot_linear_oscillator_2d(
    model,
    device,
    tau,
    x1_bounds=(-1.0, 1.0),
    x2_bounds=(-1.0, 1.0),
    nx=201,
    chunk_size=4096,
    scale_to_minus1_1=False,
    T=1.0,
    scale_time_to_01=True,
    threshold=0.25,
    show_terminal_target=True,
    title=None,
    show_heatmap=True,
    save_path=None,
    dpi=300,
    show=False,
):
    model_was_training = model.training
    model.eval()

    x1 = np.linspace(x1_bounds[0], x1_bounds[1], nx)
    x2 = np.linspace(x2_bounds[0], x2_bounds[1], nx)
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")

    pts_phys = np.stack([X1.reshape(-1), X2.reshape(-1)], axis=-1).astype(np.float32)

    pts_net = pts_phys.copy()
    if scale_to_minus1_1:
        lo = np.array([x1_bounds[0], x2_bounds[0]], dtype=np.float32)
        hi = np.array([x1_bounds[1], x2_bounds[1]], dtype=np.float32)
        pts_net = 2.0 * (pts_net - lo) / (hi - lo) - 1.0

    tau_net = tau / T if scale_time_to_01 else tau
    tau_col = np.full((pts_net.shape[0], 1), tau_net, dtype=np.float32)
    xt_np = np.concatenate([pts_net, tau_col], axis=-1)

    vals = []
    for start in range(0, xt_np.shape[0], chunk_size):
        end = min(start + chunk_size, xt_np.shape[0])

        xt_chunk = torch.tensor(xt_np[start:end], dtype=torch.float32, device=device)
        tau_phys_chunk = torch.full((end - start,), float(tau), dtype=torch.float32, device=device)

        V_chunk, _, _ = model.compute_value_at_xt(xt_chunk, tau_phys_chunk)
        if isinstance(V_chunk, tuple):
            V_chunk = V_chunk[0]

        vals.append(V_chunk.reshape(-1).detach().cpu().numpy())

    V = np.concatenate(vals, axis=0).reshape(nx, nx)

    fig = plt.figure(figsize=(7, 6))

    if show_heatmap:
        cf = plt.contourf(X1, X2, V, levels=40)
        plt.colorbar(cf, label="V(x, tau)")

    plt.contour(X1, X2, V, levels=[0.0], colors="red", linewidths=2.5, linestyles="-")

    if show_terminal_target:
        th = np.linspace(0, 2 * np.pi, 400)
        r = threshold
        plt.plot(r * np.cos(th), r * np.sin(th), color="black", linestyle="--", linewidth=2.0)

    handles = [Line2D([0], [0], color="red", lw=2.5, linestyle="-", label="learned V=0")]
    if show_terminal_target:
        handles.append(Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="true terminal target"))
    plt.legend(handles=handles)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(x1_bounds)
    plt.ylim(x2_bounds)
    plt.gca().set_aspect("equal")

    if title is None:
        title = f"Linear Oscillator Boundary at tau={tau:.2f}"
    plt.title(title)
    plt.tight_layout()

    if save_path is None:
        save_path = Path(f"linear_oscillator_tau_{tau:.2f}.png")
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if model_was_training:
        model.train()