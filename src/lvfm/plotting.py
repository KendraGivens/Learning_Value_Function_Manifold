import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from lvfm.helpers import compute_metrics

@torch.no_grad()
def plot_linear_oscillator_2d(
    model,
    device,
    tau,
    gt_values=None,              # shape: (nx, nx)
    x1_bounds=(-1.0, 1.0),
    x2_bounds=(-1.0, 1.0),
    nx=201,
    chunk_size=4096,
    scale_to_minus1_1=False,
    T=1.0,
    scale_time_to_01=True,
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

    V_pred = np.concatenate(vals, axis=0).reshape(nx, nx)

    gt_values = None if gt_values is None else np.asarray(gt_values)
    if gt_values is not None and gt_values.shape != (nx, nx):
        raise ValueError(f"gt_values must have shape {(nx, nx)}, got {gt_values.shape}")

    ncols = 2 if gt_values is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(12, 5) if ncols == 2 else (7, 6))
    if ncols == 1:
        axes = [axes]

    if gt_values is not None:
        vmin = min(V_pred.min(), gt_values.min())
        vmax = max(V_pred.max(), gt_values.max())
    else:
        vmin = V_pred.min()
        vmax = V_pred.max()

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    levels = np.linspace(vmin, vmax, 40)

    # Left panel: prediction
    ax = axes[0]
    if show_heatmap:
        cf_pred = ax.contourf(X1, X2, V_pred, levels=levels, vmin=vmin, vmax=vmax)
        fig.colorbar(cf_pred, ax=ax, label="V(x, tau)")

    ax.contour(X1, X2, V_pred, levels=[0.0], colors="red", linewidths=2.5, linestyles="-")

    if gt_values is not None:
        ax.contour(X1, X2, gt_values, levels=[0.0], colors="black", linewidths=2.5, linestyles="--")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(x1_bounds)
    ax.set_ylim(x2_bounds)
    ax.set_aspect("equal")
    ax.set_title(f"Predicted at tau={tau:.2f}")

    # Right panel: ground truth
    if gt_values is not None:
        ax = axes[1]
        if show_heatmap:
            cf_gt = ax.contourf(X1, X2, gt_values, levels=levels, vmin=vmin, vmax=vmax)
            fig.colorbar(cf_gt, ax=ax, label="V(x, tau)")

        ax.contour(X1, X2, gt_values, levels=[0.0], colors="black", linewidths=2.5, linestyles="--")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(x1_bounds)
        ax.set_ylim(x2_bounds)
        ax.set_aspect("equal")
        ax.set_title(f"Ground Truth at tau={tau:.2f}")

    handles = [Line2D([0], [0], color="red", lw=2.5, linestyle="-", label="learned V=0")]
    if gt_values is not None:
        handles.append(Line2D([0], [0], color="black", lw=2.5, linestyle="--", label="ground-truth V=0"))

    if title is None:
        title = f"Predicted vs Ground Truth at tau={tau:.2f}"
    fig.suptitle(title, y=0.99)
    
    if gt_values is not None:
        mae, iou = compute_metrics(V_pred, gt_values)
        metric_text = f"MAE: {mae:.4e}    BRT Overlap: {iou:.4f}"
        fig.text(
            0.5,
            0.93,
            metric_text,
            ha="center",
            va="center",
        )
    
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.915),
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    if save_path is None:
        save_path = Path(f"linear_oscillator_compare_tau_{tau:.2f}.png")
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