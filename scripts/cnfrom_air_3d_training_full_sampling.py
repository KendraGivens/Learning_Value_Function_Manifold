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
from types import SimpleNamespace
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from lvfm.datasets import Air3DDataset
from lvfm.residuals import Air3DResidual
from lvfm.managers import LossManager
from lvfm.models import Decoder, PNODE_MLP, PNODE_Siren, INR_PNODE
from lvfm.hj_solvers import solve_air3d_relative
from lvfm.plotting import plot_air3d_pnode_slice, build_air3d_gt_slice
from lvfm.training import build_residual_distribution_pool, sample_from_residual_distribution, RARDAnchorSet, replace_interior_points

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
    train_dataset = Air3DDataset(
        num_batches=cfg.num_batches,
        num_interior=cfg.num_interior,
        num_terminal=cfg.num_terminal,
        T=cfg.T,
        num_unique_taus=cfg.num_unique_taus,
        x_bounds=cfg.x_bounds,
        y_bounds=cfg.y_bounds,
        psi_bounds=cfg.psi_bounds,
        tau_max=0.0,
        scale_to_minus1_1=cfg.scale_to_minus1_1,
        scale_time_to_0_1=cfg.scale_time_to_01,
        efficient=cfg.efficent
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
    )    
    
    return train_dataset, train_loader

def create_residual(cfg):
    residual = Air3DResidual(
        vp=cfg.vp,
        ve=cfg.ve,
        control_bound=cfg.u_bound,
        disturbance_bound=cfg.d_bound,
        radius=cfg.beta,
        T=cfg.T,
        x_bounds=cfg.x_bounds,
        y_bounds=cfg.y_bounds,
        psi_bounds=cfg.psi_bounds,
        scale_to_minus1_1=cfg.scale_to_minus1_1,
        scale_time_to_01=cfg.scale_time_to_01,
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

    model.causal_loss = getattr(cfg, "causal_loss", False)
    model.causal_chunks = getattr(cfg, "causal_chunks", 16)
    model.causal_eps = getattr(cfg, "causal_eps", 1.0)

    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--device", type=int)
    args = p.parse_args()
    
    cfg_path = Path("configs") / "air_3d" / f"{args.cfg}.yaml"
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
    run_dir = Path("runs/air_3d") / run_name
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

    plot_air3d_pnode_slice(
        model=model,
        device=cfg.device,
        tau=0.0,
        psi_slice=cfg.psi_slice,
        gt_values=gt_slice,
        x_bounds=cfg.x_bounds,
        y_bounds=cfg.y_bounds,
        psi_bounds=cfg.psi_bounds,
        nx=cfg.x_discretization,
        chunk_size=4096,
        scale_to_minus1_1=cfg.scale_to_minus1_1,
        T=cfg.T,
        scale_time_to_01=cfg.scale_time_to_01,
        title=f"Air3D slice at tau=0.00, psi={cfg.psi_slice:.2f}",
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
    adaptive_method = getattr(cfg, "adaptive_method", "none").lower()
    rad_pool = None
    rar_d_anchors = RARDAnchorSet(
        max_anchors=getattr(cfg, "rar_d_max_anchors", None)
    )
    
    while step < total_steps:
        for batch in train_loader:
            for opt in optimizers.values():
                opt.zero_grad()
    
            if adaptive_method == "rad":
                if rad_pool is None or step % cfg.adaptive_every == 0:
                    rad_pool = build_residual_distribution_pool(
                        model=model,
                        residual=residual,
                        train_dataset=train_dataset,
                        device=cfg.device,
                        num_candidates=cfg.num_rad_candidates,
                        chunk_size=getattr(cfg, "adaptive_chunk_size", 2048),
                        rad_k=getattr(cfg, "rad_k", 1.0),
                        rad_c=getattr(cfg, "rad_c", 1.0),
                        rad_eps=getattr(cfg, "rad_eps", 1e-8),
                        independent_tau=getattr(cfg, "rad_independent_tau", True),
                    )

                
                sampled = sample_from_residual_distribution(
                    pool=rad_pool,
                    num_samples=cfg.num_interior,
                    replace=False,
                )
    
                batch = replace_interior_points(batch, sampled)
    
            elif adaptive_method == "rar_d":
                if step % cfg.adaptive_every == 0:
                    rad_pool = build_residual_distribution_pool(
                        model=model,
                        residual=residual,
                        train_dataset=train_dataset,
                        device=cfg.device,
                        num_candidates=cfg.num_rad_candidates,
                        chunk_size=getattr(cfg, "adaptive_chunk_size", 2048),
                        rad_k=getattr(cfg, "rad_k", 1.0),
                        rad_c=getattr(cfg, "rad_c", 1.0),
                        rad_eps=getattr(cfg, "rad_eps", 1e-8),
                        independent_tau=getattr(cfg, "rad_independent_tau", True),
                    )
    
                    sampled = sample_from_residual_distribution(
                        pool=rad_pool,
                        num_samples=getattr(cfg, "rar_d_add", 1),
                        replace=False,
                    )
    
                    rar_d_anchors.add(sampled)
    
                    print(
                        f"RAR-D step={step}: "
                        f"anchors={rar_d_anchors.num_anchors()}, "
                        f"mean |R|={rad_pool['scores'].mean().item():.4e}, "
                        f"max |R|={rad_pool['scores'].max().item():.4e}"
                    )
    
                batch = rar_d_anchors.append_to_batch(batch)
    
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
    
                    plot_air3d_pnode_slice(
                        model=model,
                        device=cfg.device,
                        tau=eval_tau,
                        psi_slice=cfg.psi_slice,
                        gt_values=gt_slice,
                        x_bounds=cfg.x_bounds,
                        y_bounds=cfg.y_bounds,
                        psi_bounds=cfg.psi_bounds,
                        nx=cfg.x_discretization,
                        chunk_size=4096,
                        scale_to_minus1_1=cfg.scale_to_minus1_1,
                        T=cfg.T,
                        scale_time_to_01=cfg.scale_time_to_01,
                        title=f"Air3D slice at tau={eval_tau:.2f}, psi={cfg.psi_slice:.2f}",
                        save_path=plot_dir / f"compare_tau_{eval_tau:.2f}.png",
                    )
    
            if step >= total_steps:
                break
            
if __name__ == "__main__":
    main()
