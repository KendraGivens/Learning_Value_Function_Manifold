import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
from pathlib import Path
import shutil
import torch
import yaml
import random
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from types import SimpleNamespace
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from lvfm.datasets import Air6DJointDataset
from lvfm.residuals import Air6DJointResidual
from lvfm.managers import LossManager
from lvfm.models import Decoder, PNODE_MLP, PNODE_Siren, INR_PNODE
from lvfm.hj_solvers import solve_air3d_relative
from lvfm.plotting import plot_air3d_pnode_slice, build_air3d_gt_slice

def to_namespace(d):
    return SimpleNamespace(**d)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_checkpoint(path, model, optimizers, step=None, tau_max=None, cfg=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": {k: opt.state_dict() for k, opt in optimizers.items()},
        "step": step,
        "tau_max": tau_max
    }
    if cfg is not None: 
        ckpt["cfg"] = vars(cfg) if hasattr(cfg, "__dict__") else cfg

    torch.save(ckpt, path)

def create_dataloader(cfg):    
    train_dataset = Air6DJointDataset(
        num_batches=cfg.num_batches,
        num_interior=cfg.num_interior,
        num_terminal=cfg.num_terminal,
        T=cfg.T,
        num_unique_taus=cfg.num_unique_taus,
        x_bounds=cfg.x_bounds,
        y_bounds=cfg.y_bounds,
        theta_bounds=cfg.psi_bounds,
        tau_max=0.0,
        scale_to_minus1_1=cfg.scale_to_minus1_1,
        scale_time_to_0_1=cfg.scale_time_to_01,
        efficient=getattr(cfg, "efficient", getattr(cfg, "efficent", False)),
        angle_alpha_factor=getattr(cfg, "angle_alpha_factor", 1.2),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
    )    
    
    return train_dataset, train_loader

def create_residual(cfg):
    residual = Air6DJointResidual(
        vp=cfg.vp,
        ve=cfg.ve,
        control_bound=cfg.u_bound,
        disturbance_bound=cfg.d_bound,
        radius=cfg.beta,
        T=cfg.T,
        x_bounds=cfg.x_bounds,
        y_bounds=cfg.y_bounds,
        theta_bounds=cfg.psi_bounds,
        scale_to_minus1_1=cfg.scale_to_minus1_1,
        scale_time_to_01=cfg.scale_time_to_01,
        angle_alpha_factor=getattr(cfg, "angle_alpha_factor", 1.2),
    )
    return residual

def create_model(cfg, residual):
    loss_manager = LossManager(
        residual=residual,
        loss_weights={"terminal": 1.0, "pinn": 1.0},
    )

    decoder = Decoder(
        hidden_dim=cfg.decoder_hidden_dim,
        latent_dim=cfg.latent_dim,
        coordinate_dim=cfg.coordinate_dim,
        out_dim=1,
        num_layers=cfg.decoder_num_layers,
        net_type=cfg.net_type,
        input_scale=cfg.input_scale,
    )

    if cfg.pnode == "siren":
        pnode = PNODE_Siren(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.pnode_hidden_dim,
        )
    elif cfg.pnode == "mlp":
        pnode = PNODE_MLP(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.pnode_hidden_dim,
        )

    model = INR_PNODE(
        latent_dim=cfg.latent_dim,
        decoder=decoder,
        pnode=pnode,
        loss_manager=loss_manager,
        value_var=0.5,
        value_normto=0.02,
        ode_solver=odeint,
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
        device=cfg.device,
    ).to(cfg.device)

    return model

def scale_air6d_states_to_net(x_phys, cfg):
    x = torch.tensor(x_phys, dtype=torch.float32)

    if getattr(cfg, "scale_to_minus1_1", True):
        angle_scale = getattr(cfg, "angle_alpha_factor", 1.2) * np.pi

        # xp, yp
        x[:, 0] = 2.0 * (x[:, 0] - cfg.x_bounds[0]) / (
            cfg.x_bounds[1] - cfg.x_bounds[0]
        ) - 1.0

        x[:, 1] = 2.0 * (x[:, 1] - cfg.y_bounds[0]) / (
            cfg.y_bounds[1] - cfg.y_bounds[0]
        ) - 1.0

        # theta_p
        x[:, 2] = x[:, 2] / angle_scale

        # xe, ye
        x[:, 3] = 2.0 * (x[:, 3] - cfg.x_bounds[0]) / (
            cfg.x_bounds[1] - cfg.x_bounds[0]
        ) - 1.0

        x[:, 4] = 2.0 * (x[:, 4] - cfg.y_bounds[0]) / (
            cfg.y_bounds[1] - cfg.y_bounds[0]
        ) - 1.0

        # theta_e
        x[:, 5] = x[:, 5] / angle_scale

    return x


def evaluate_air6d_projected_slice(
    model,
    cfg,
    tau,
    psi_slice,
    nx=101,
    device=None,
    chunk_size=4096,
):
    if device is None:
        device = cfg.device

    xr = np.linspace(cfg.x_bounds[0], cfg.x_bounds[1], nx)
    yr = np.linspace(cfg.y_bounds[0], cfg.y_bounds[1], nx)

    X, Y = np.meshgrid(xr, yr, indexing="xy")

    # Projection:
    # [xp, yp, theta_p, xe, ye, theta_e]
    # = [xr, yr, psi, 0, 0, 0]
    pts_phys = np.stack(
        [
            X.reshape(-1),
            Y.reshape(-1),
            np.full(X.size, psi_slice),
            np.zeros(X.size),
            np.zeros(X.size),
            np.zeros(X.size),
        ],
        axis=-1,
    ).astype(np.float32)

    x_net = scale_air6d_states_to_net(pts_phys, cfg)

    tau_net = torch.full((x_net.shape[0], 1), float(tau), dtype=torch.float32)
    if getattr(cfg, "scale_time_to_01", True):
        tau_net = tau_net / cfg.T

    xt = torch.cat([x_net, tau_net], dim=-1).to(device)

    vals = []
    model.eval()

    for start in range(0, xt.shape[0], chunk_size):
        end = min(start + chunk_size, xt.shape[0])

        xt_chunk = xt[start:end]
        tau_phys_chunk = torch.full(
            (end - start,),
            float(tau),
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            V_chunk = model.compute_value_at_xt(xt_chunk, tau_phys_chunk)[0]

        vals.append(V_chunk.reshape(-1).detach().cpu().numpy())

    V = np.concatenate(vals, axis=0).reshape(nx, nx)

    return X, Y, V

def compute_iou(V_pred, V_gt):
    pred_set = V_pred <= 0.0
    gt_set = V_gt <= 0.0

    inter = np.logical_and(pred_set, gt_set).sum()
    union = np.logical_or(pred_set, gt_set).sum()

    return inter / max(union, 1)


def plot_air6d_projected_comparison(
    X,
    Y,
    V_pred,
    V_gt,
    tau,
    psi_slice,
    save_path,
    title=None,
):
    mae = np.mean(np.abs(V_pred - V_gt))
    iou = compute_iou(V_pred, V_gt)

    vmin = min(np.nanmin(V_pred), np.nanmin(V_gt))
    vmax = max(np.nanmax(V_pred), np.nanmax(V_gt))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    levels = np.linspace(vmin, vmax, 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    ax = axes[0]
    cf = ax.contourf(X, Y, V_pred, levels=levels)
    ax.contour(X, Y, V_pred, levels=[0.0], colors="red", linewidths=2.0)
    ax.contour(X, Y, V_gt, levels=[0.0], colors="black", linestyles="--", linewidths=2.0)
    ax.set_title(f"6D projected prediction\nτ={tau:.2f}, ψ={psi_slice:.2f}")
    ax.set_xlabel(r"$x_{rel}$")
    ax.set_ylabel(r"$y_{rel}$")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax)

    ax = axes[1]
    cf = ax.contourf(X, Y, V_gt, levels=levels)
    ax.contour(X, Y, V_gt, levels=[0.0], colors="black", linewidths=2.0)
    ax.set_title("3D relative ground truth")
    ax.set_xlabel(r"$x_{rel}$")
    ax.set_ylabel(r"$y_{rel}$")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax)

    ax = axes[2]
    err = np.abs(V_pred - V_gt)
    cf = ax.contourf(X, Y, err, levels=60)
    ax.contour(X, Y, V_pred, levels=[0.0], colors="red", linewidths=2.0)
    ax.contour(X, Y, V_gt, levels=[0.0], colors="black", linestyles="--", linewidths=2.0)
    ax.set_title("Absolute error")
    ax.set_xlabel(r"$x_{rel}$")
    ax.set_ylabel(r"$y_{rel}$")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax)

    if title is None:
        title = (
            f"Air6D projected to relative coordinates | "
            f"MAE={mae:.4e}, IoU={iou:.4f}"
        )

    fig.suptitle(title)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return mae, iou

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--device", type=int)
    args = p.parse_args()
    
    cfg_path = Path("configs") / "air_6d" / f"{args.cfg}.yaml"
    cfg = load_yaml(cfg_path)
    cfg = to_namespace(cfg)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_idx = 0 if args.device is None else args.device
    cfg.device = torch.device(f"cuda:{device_idx}")

    run_name = args.cfg
    run_dir = Path("runs/air_6d") / run_name
    ckpt_dir = run_dir / "ckpts"
    plot_dir = run_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, run_dir / "config.yaml")
            
    train_dataset, train_loader = create_dataloader(cfg)
    residual = create_residual(cfg)
    model = create_model(cfg, residual)
    optimizers = model.create_optimizers(lr=cfg.lr)

    gt_value_function = solve_air3d_relative(
            tau_steps=cfg.tau_schedule,
            xr_discretization=cfg.x_discretization,
            yr_discretization=cfg.y_discretization,
            theta_discretization=cfg.psi_discretization,
            xr_bounds=cfg.x_bounds,
            yr_bounds=cfg.y_bounds,
            theta_bounds=cfg.psi_bounds,
            vp=cfg.vp,
            ve=cfg.ve,
            u_bound=cfg.u_bound,
            d_bound=cfg.d_bound,
            radius=cfg.beta,
        )
    
    # pretraining
    train_dataset.set_tau_max(0.0)
    model.loss_manager.loss_weights = {"terminal": 1.0, "pinn": 0.0}
    step = 0
    while step < cfg.pretrain_steps:
        for batch in train_loader:
            for opt in optimizers.values():
                opt.zero_grad()

            results = model.compute_losses(batch, mode="train")
            results["loss_train"].backward()

            for opt in optimizers.values():
                opt.step()
            step += 1
            if step >= cfg.pretrain_steps:
                break

    save_checkpoint(ckpt_dir/"pretrain.pt", model, optimizers, step=step, tau_max=0.0, cfg=cfg)
    gt_slice = build_air3d_gt_slice(
        gt_values_3d=np.asarray(gt_value_function[0]),
        psi_bounds=cfg.psi_bounds,
        psi_slice=cfg.psi_slice,
    )

    X, Y, V_pred = evaluate_air6d_projected_slice(
        model=model,
        cfg=cfg,
        tau=0.0,
        psi_slice=cfg.psi_slice,
        nx=cfg.x_discretization,
        device=cfg.device,
    )
    
    mae, iou = plot_air6d_projected_comparison(
        X=X,
        Y=Y,
        V_pred=V_pred,
        V_gt=gt_slice,
        tau=0.0,
        psi_slice=cfg.psi_slice,
        save_path=plot_dir / "compare_tau_0.00.png",
    )
    
    model.loss_manager.loss_weights = {"terminal": 0.0, "pinn": 1.0}
    
    train_dataset.set_tau_max(cfg.T)
    
    if hasattr(cfg, "total_steps"):
        total_steps = cfg.total_steps
    else:
        total_steps = cfg.steps_per_stage * len(cfg.tau_schedule)
    
    if hasattr(cfg, "save_freq"):
        save_freq = cfg.save_freq
    else:
        save_freq = cfg.steps_per_stage
    
    step = 0
    while step < total_steps:
        for batch in train_loader:
            for opt in optimizers.values():
                opt.zero_grad()
    
            results = model.compute_losses(batch, mode="train")
            results["loss_train"].backward()

            for opt in optimizers.values():
                opt.step()
    
            step += 1
    
            if step % save_freq == 0 or step == total_steps:
                print(f"step={step}/{total_steps}")
    
                save_checkpoint(
                    ckpt_dir / f"step_{step:06d}.pt",
                    model,
                    optimizers,
                    step=step,
                    tau_max=cfg.T,
                    cfg=cfg,
                )
    
                for i, eval_tau in enumerate(cfg.tau_schedule):
                    gt_slice = build_air3d_gt_slice(
                        gt_values_3d=np.asarray(gt_value_function[i + 1]),
                        psi_bounds=cfg.psi_bounds,
                        psi_slice=cfg.psi_slice,
                    )
                
                    X, Y, V_pred = evaluate_air6d_projected_slice(
                        model=model,
                        cfg=cfg,
                        tau=eval_tau,
                        psi_slice=cfg.psi_slice,
                        nx=cfg.x_discretization,
                        device=cfg.device,
                    )
                
                    mae, iou = plot_air6d_projected_comparison(
                        X=X,
                        Y=Y,
                        V_pred=V_pred,
                        V_gt=gt_slice,
                        tau=eval_tau,
                        psi_slice=cfg.psi_slice,
                        save_path=plot_dir / f"compare_tau_{eval_tau:.2f}.png",
                    )
                
                    print(f"tau={eval_tau:.2f} projected MAE={mae:.4e}, IoU={iou:.4f}")    
                    
            if step >= total_steps:
                break
            
if __name__ == "__main__":
    main()
