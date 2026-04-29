import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from lvfm.helpers import compute_metrics
from scipy.interpolate import RegularGridInterpolator
import math

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
    
        V_chunk = model.compute_value_at_xt(xt_chunk, tau_phys_chunk)[0]
        vals.append(V_chunk.reshape(-1).detach().cpu().numpy())

    V_pred = np.concatenate(vals, axis=0).reshape(nx, nx)

    gt_values = None if gt_values is None else np.asarray(gt_values)
    if gt_values is not None and gt_values.shape != (nx, nx):
        raise ValueError(f"gt_values must have shape {(nx, nx)}, got {gt_values.shape}")

    abs_error = None
    if gt_values is not None:
        abs_error = np.abs(V_pred - gt_values)

    ncols = 3 if gt_values is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(18, 5) if ncols == 3 else (7, 6))
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

    if gt_values is not None:
        # Middle panel: ground truth
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

        # Right panel: absolute error
        ax = axes[2]
        err_levels = np.linspace(0.0, max(abs_error.max(), 1e-8), 40)
        cf_err = ax.contourf(X1, X2, abs_error, levels=err_levels)
        fig.colorbar(cf_err, ax=ax, label=r"$|V_{\mathrm{pred}} - V_{\mathrm{gt}}|$")

        ax.contour(X1, X2, V_pred, levels=[0.0], colors="red", linewidths=2.0, linestyles="-")
        ax.contour(X1, X2, gt_values, levels=[0.0], colors="black", linewidths=2.0, linestyles="--")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(x1_bounds)
        ax.set_ylim(x2_bounds)
        ax.set_aspect("equal")
        ax.set_title("Absolute Error")

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

@torch.no_grad()
def plot_linear_oscillator_2d_deepreach(
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

    # Model expects xt = [x1, x2, tau]
    xt_np = np.concatenate([pts_net, tau_col], axis=-1)

    vals = []
    for start in range(0, xt_np.shape[0], chunk_size):
        end = min(start + chunk_size, xt_np.shape[0])
        xt_chunk = torch.tensor(xt_np[start:end], dtype=torch.float32, device=device)

        V_chunk = model(xt_chunk).squeeze(-1)
        vals.append(V_chunk.detach().cpu().numpy())

    V_pred = np.concatenate(vals, axis=0).reshape(nx, nx)

    gt_values = None if gt_values is None else np.asarray(gt_values)
    if gt_values is not None and gt_values.shape != (nx, nx):
        raise ValueError(f"gt_values must have shape {(nx, nx)}, got {gt_values.shape}")

    abs_error = None
    if gt_values is not None:
        abs_error = np.abs(V_pred - gt_values)

    ncols = 3 if gt_values is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(18, 5) if ncols == 3 else (7, 6))
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

    if gt_values is not None:
        # Middle panel: ground truth
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

        # Right panel: absolute error
        ax = axes[2]
        err_levels = np.linspace(0.0, max(abs_error.max(), 1e-8), 40)
        cf_err = ax.contourf(X1, X2, abs_error, levels=err_levels)
        fig.colorbar(cf_err, ax=ax, label=r"$|V_{\mathrm{pred}} - V_{\mathrm{gt}}|$")

        ax.contour(X1, X2, V_pred, levels=[0.0], colors="red", linewidths=2.0, linestyles="-")
        ax.contour(X1, X2, gt_values, levels=[0.0], colors="black", linewidths=2.0, linestyles="--")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(x1_bounds)
        ax.set_ylim(x2_bounds)
        ax.set_aspect("equal")
        ax.set_title("Absolute Error")

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
        save_path = Path(f"linear_oscillator_deepreach_tau_{tau:.2f}.png")
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

def build_air3d_gt_slice(gt_values_3d, psi_bounds, psi_slice):
    npsi = gt_values_3d.shape[2]
    psi_axis = np.linspace(psi_bounds[0], psi_bounds[1], npsi)
    psi_idx = np.argmin(np.abs(psi_axis - psi_slice))
    return np.asarray(gt_values_3d[:, :, psi_idx]).T

@torch.no_grad()
def plot_air3d_pnode_slice(
    model,
    device,
    tau,
    psi_slice,
    gt_values=None,
    x_bounds=(-1.0, 1.0),
    y_bounds=(-1.0, 1.0),
    psi_bounds=(-3.14159265, 3.14159265),
    nx=101,
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

    x = np.linspace(x_bounds[0], x_bounds[1], nx)
    y = np.linspace(y_bounds[0], y_bounds[1], nx)
    X, Y = np.meshgrid(x, y, indexing="xy")

    psi = np.full((X.size, 1), psi_slice, dtype=np.float32)
    pts_phys = np.concatenate(
        [X.reshape(-1, 1).astype(np.float32), Y.reshape(-1, 1).astype(np.float32), psi],
        axis=-1,
    )

    pts_net = pts_phys.copy()
    if scale_to_minus1_1:
        lo = np.array([x_bounds[0], y_bounds[0], psi_bounds[0]], dtype=np.float32)
        hi = np.array([x_bounds[1], y_bounds[1], psi_bounds[1]], dtype=np.float32)
        pts_net = 2.0 * (pts_net - lo) / (hi - lo) - 1.0

    tau_net = tau / T if scale_time_to_01 else tau
    tau_col = np.full((pts_net.shape[0], 1), tau_net, dtype=np.float32)
    xt_np = np.concatenate([pts_net, tau_col], axis=-1)

    vals = []
    for start in range(0, xt_np.shape[0], chunk_size):
        end = min(start + chunk_size, xt_np.shape[0])
        xt_chunk = torch.tensor(xt_np[start:end], dtype=torch.float32, device=device)

        # physical tau for PNODE latent evolution
        tau_chunk = torch.full((end - start,), tau, dtype=torch.float32, device=device)

        V_chunk, _, _ = model.compute_value_at_xt(xt_chunk, tau_chunk)
        vals.append(V_chunk.detach().cpu().numpy())

    V_pred = np.concatenate(vals, axis=0).reshape(nx, nx)

    gt_values = None if gt_values is None else np.asarray(gt_values)
    if gt_values is not None and gt_values.shape != (nx, nx):
        raise ValueError(f"gt_values must have shape {(nx, nx)}, got {gt_values.shape}")

    abs_error = None
    if gt_values is not None:
        abs_error = np.abs(V_pred - gt_values)

    ncols = 3 if gt_values is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(18, 5) if ncols == 3 else (7, 6))
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

    ax = axes[0]
    if show_heatmap:
        cf_pred = ax.contourf(X, Y, V_pred, levels=levels, vmin=vmin, vmax=vmax)
        fig.colorbar(cf_pred, ax=ax, label="V(x, y, psi, tau)")
    ax.contour(X, Y, V_pred, levels=[0.0], colors="red", linewidths=2.5, linestyles="-")
    if gt_values is not None:
        ax.contour(X, Y, gt_values, levels=[0.0], colors="black", linewidths=2.5, linestyles="--")
    ax.set_xlabel("$x_{rel}$")
    ax.set_ylabel("$y_{rel}$")
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect("equal")
    ax.set_title(f"Predicted at tau={tau:.2f}, psi={psi_slice:.2f}")

    if gt_values is not None:
        ax = axes[1]
        if show_heatmap:
            cf_gt = ax.contourf(X, Y, gt_values, levels=levels, vmin=vmin, vmax=vmax)
            fig.colorbar(cf_gt, ax=ax, label="V(x, y, psi, tau)")
        ax.contour(X, Y, gt_values, levels=[0.0], colors="black", linewidths=2.5, linestyles="--")
        ax.set_xlabel("$x_{rel}$")
        ax.set_ylabel("$y_{rel}$")
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect("equal")
        ax.set_title(f"Ground Truth at tau={tau:.2f}, psi={psi_slice:.2f}")

        ax = axes[2]
        err_levels = np.linspace(0.0, max(abs_error.max(), 1e-8), 40)
        cf_err = ax.contourf(X, Y, abs_error, levels=err_levels)
        fig.colorbar(cf_err, ax=ax, label=r"$|V_{\mathrm{pred}} - V_{\mathrm{gt}}|$")
        ax.contour(X, Y, V_pred, levels=[0.0], colors="red", linewidths=2.0, linestyles="-")
        ax.contour(X, Y, gt_values, levels=[0.0], colors="black", linewidths=2.0, linestyles="--")
        ax.set_xlabel("$x_{rel}$")
        ax.set_ylabel("$y_{rel}$")
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect("equal")
        ax.set_title("Absolute Error")

    handles = [Line2D([0], [0], color="red", lw=2.5, linestyle="-", label="learned V=0")]
    if gt_values is not None:
        handles.append(Line2D([0], [0], color="black", lw=2.5, linestyle="--", label="ground-truth V=0"))

    if title is None:
        title = f"Air3D slice at tau={tau:.2f}, psi={psi_slice:.2f}"
    fig.suptitle(title, y=0.99)

    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.915))
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    if save_path is None:
        save_path = Path(f"air3d_pnode_tau_{tau:.2f}_psi_{psi_slice:.2f}.png")
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

@torch.no_grad()
def plot_air3d_pnode_slice(
    model,
    device,
    tau,
    psi_slice,
    gt_values=None,
    x_bounds=(-1.0, 1.0),
    y_bounds=(-1.0, 1.0),
    psi_bounds=(-3.14159265, 3.14159265),
    nx=101,
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

    x = np.linspace(x_bounds[0], x_bounds[1], nx)
    y = np.linspace(y_bounds[0], y_bounds[1], nx)
    X, Y = np.meshgrid(x, y, indexing="xy")

    psi = np.full((X.size, 1), psi_slice, dtype=np.float32)
    pts_phys = np.concatenate(
        [X.reshape(-1, 1).astype(np.float32), Y.reshape(-1, 1).astype(np.float32), psi],
        axis=-1,
    )

    pts_net = pts_phys.copy()
    if scale_to_minus1_1:
        # x and y keep min-max scaling
        pts_net[:, 0] = 2.0 * (pts_phys[:, 0] - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1.0
        pts_net[:, 1] = 2.0 * (pts_phys[:, 1] - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) - 1.0

        # psi must match training-time scaling
        pts_net[:, 2] = pts_phys[:, 2] / (1.2 * math.pi)

    tau_net = tau / T if scale_time_to_01 else tau
    tau_col = np.full((pts_net.shape[0], 1), tau_net, dtype=np.float32)
    xt_np = np.concatenate([pts_net, tau_col], axis=-1)

    vals = []
    for start in range(0, xt_np.shape[0], chunk_size):
        end = min(start + chunk_size, xt_np.shape[0])
        xt_chunk = torch.tensor(xt_np[start:end], dtype=torch.float32, device=device)

        # physical tau for PNODE latent evolution
        tau_chunk = torch.full((end - start,), tau, dtype=torch.float32, device=device)

        # compute_value_at_xt now returns 4 values
        V_chunk = model.compute_value_at_xt(xt_chunk, tau_chunk)[0]
        vals.append(V_chunk.detach().cpu().numpy())

    V_pred = np.concatenate(vals, axis=0).reshape(nx, nx)

    gt_values = None if gt_values is None else np.asarray(gt_values)
    if gt_values is not None and gt_values.shape != (nx, nx):
        raise ValueError(f"gt_values must have shape {(nx, nx)}, got {gt_values.shape}")

    abs_error = None
    if gt_values is not None:
        abs_error = np.abs(V_pred - gt_values)

    ncols = 3 if gt_values is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(18, 5) if ncols == 3 else (7, 6))
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

    ax = axes[0]
    if show_heatmap:
        cf_pred = ax.contourf(X, Y, V_pred, levels=levels, vmin=vmin, vmax=vmax)
        fig.colorbar(cf_pred, ax=ax, label="V(x, y, psi, tau)")
    ax.contour(X, Y, V_pred, levels=[0.0], colors="red", linewidths=2.5, linestyles="-")
    if gt_values is not None:
        ax.contour(X, Y, gt_values, levels=[0.0], colors="black", linewidths=2.5, linestyles="--")
    ax.set_xlabel("$x_{rel}$")
    ax.set_ylabel("$y_{rel}$")
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect("equal")
    ax.set_title(f"Predicted at tau={tau:.2f}, psi={psi_slice:.2f}")

    if gt_values is not None:
        ax = axes[1]
        if show_heatmap:
            cf_gt = ax.contourf(X, Y, gt_values, levels=levels, vmin=vmin, vmax=vmax)
            fig.colorbar(cf_gt, ax=ax, label="V(x, y, psi, tau)")
        ax.contour(X, Y, gt_values, levels=[0.0], colors="black", linewidths=2.5, linestyles="--")
        ax.set_xlabel("$x_{rel}$")
        ax.set_ylabel("$y_{rel}$")
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect("equal")
        ax.set_title(f"Ground Truth at tau={tau:.2f}, psi={psi_slice:.2f}")

        ax = axes[2]
        err_levels = np.linspace(0.0, max(abs_error.max(), 1e-8), 40)
        cf_err = ax.contourf(X, Y, abs_error, levels=err_levels)
        fig.colorbar(cf_err, ax=ax, label=r"$|V_{\mathrm{pred}} - V_{\mathrm{gt}}|$")
        ax.contour(X, Y, V_pred, levels=[0.0], colors="red", linewidths=2.0, linestyles="-")
        ax.contour(X, Y, gt_values, levels=[0.0], colors="black", linewidths=2.0, linestyles="--")
        ax.set_xlabel("$x_{rel}$")
        ax.set_ylabel("$y_{rel}$")
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect("equal")
        ax.set_title("Absolute Error")

    handles = [Line2D([0], [0], color="red", lw=2.5, linestyle="-", label="learned V=0")]
    if gt_values is not None:
        handles.append(Line2D([0], [0], color="black", lw=2.5, linestyle="--", label="ground-truth V=0"))

    if title is None:
        title = f"Air3D slice at tau={tau:.2f}, psi={psi_slice:.2f}"
    fig.suptitle(title, y=0.99)

    if gt_values is not None:
        mae, iou = compute_metrics(V_pred, gt_values)
        metric_text = f"MAE: {mae:.4e}    BRT Overlap: {iou:.4f}"
        fig.text(0.5, 0.93, metric_text, ha="center", va="center")

    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.915))
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    if save_path is None:
        save_path = Path(f"air3d_deepreach_tau_{tau:.2f}_psi_{psi_slice:.2f}.png")
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


@torch.no_grad()
def plot_joint_dubins_6d_slice(
    model,
    device,
    tau,
    varying_dims=(0, 1),          # e.g. vary xp, yp
    fixed_state_phys=None,        # length-6 array for the remaining coordinates
    state_bounds=None,            # list of 6 (low, high) tuples
    gt_values=None,               # optional 2D array on same slice
    nx=201,
    chunk_size=4096,
    scale_to_minus1_1=True,
    T=1.0,
    scale_time_to_01=True,
    title=None,
    save_path=None,
    show=False,
):
    """
    Plots a 2D slice of a 6D value function by varying two coordinates
    and fixing the other four.
    """
    assert len(varying_dims) == 2
    assert state_bounds is not None and len(state_bounds) == 6

    if fixed_state_phys is None:
        fixed_state_phys = np.zeros(6, dtype=np.float32)
    else:
        fixed_state_phys = np.asarray(fixed_state_phys, dtype=np.float32)
        assert fixed_state_phys.shape == (6,)

    model_was_training = model.training
    model.eval()

    d1, d2 = varying_dims
    x1 = np.linspace(state_bounds[d1][0], state_bounds[d1][1], nx)
    x2 = np.linspace(state_bounds[d2][0], state_bounds[d2][1], nx)
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")

    states_phys = np.repeat(fixed_state_phys[None, :], nx * nx, axis=0)
    states_phys[:, d1] = X1.reshape(-1)
    states_phys[:, d2] = X2.reshape(-1)

    states_net = states_phys.copy()
    if scale_to_minus1_1:
        lo = np.array([b[0] for b in state_bounds], dtype=np.float32)
        hi = np.array([b[1] for b in state_bounds], dtype=np.float32)
        states_net = 2.0 * (states_net - lo) / (hi - lo) - 1.0

    tau_net = tau / T if scale_time_to_01 else tau
    tau_col = np.full((states_net.shape[0], 1), tau_net, dtype=np.float32)
    xt_np = np.concatenate([states_net, tau_col], axis=-1)

    vals = []
    for start in range(0, xt_np.shape[0], chunk_size):
        end = min(start + chunk_size, xt_np.shape[0])
        xt_chunk = torch.tensor(xt_np[start:end], dtype=torch.float32, device=device)
        V_chunk = model(xt_chunk).squeeze(-1)
        vals.append(V_chunk.detach().cpu().numpy())

    V_pred = np.concatenate(vals, axis=0).reshape(nx, nx)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    levels = 40
    cf = ax.contourf(X1, X2, V_pred, levels=levels)
    fig.colorbar(cf, ax=ax, label="V")

    ax.contour(X1, X2, V_pred, levels=[0.0], colors="red", linewidths=2.5)

    if gt_values is not None:
        gt_values = np.asarray(gt_values)
        if gt_values.shape != (nx, nx):
            raise ValueError(f"gt_values must have shape {(nx, nx)}, got {gt_values.shape}")
        ax.contour(X1, X2, gt_values, levels=[0.0], colors="black", linewidths=2.0, linestyles="--")

    ax.set_xlabel(f"state[{d1}]")
    ax.set_ylabel(f"state[{d2}]")
    ax.set_title(title or f"6D slice at tau={tau:.2f}")
    ax.set_aspect("equal")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if model_was_training:
        model.train()