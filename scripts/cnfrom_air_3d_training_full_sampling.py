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
from lvfm.datasets import Air3DEfficentDataset
from lvfm.residuals import Air3DResidual
from lvfm.managers import LossManager
from lvfm.models import Decoder, PNODE, INR_PNODE
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
    train_dataset = Air3DEfficentDataset(
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

# def create_model(cfg, residual):
#     loss_manager = LossManager(
#         residual=residual,
#         loss_weights={"terminal": 1.0, "pinn": 1.0},
#     )
    
#     decoder = Decoder(
#         hidden_dim=cfg.decoder_hidden_dim,
#         latent_dim=cfg.latent_dim,
#         coordinate_dim=cfg.coordinate_dim,
#         out_dim=1,
#         num_layers=cfg.decoder_num_layers,
#         net_type=cfg.net_type,   
#         input_scale=cfg.input_scale,
#     )
    
#     pnode = PNODE(
#         latent_dim=cfg.latent_dim,
#         hidden_dim=cfg.pnode_hidden_dim,
#     )
    
#     model = INR_PNODE(
#         latent_dim=cfg.latent_dim,
#         decoder=decoder,
#         pnode=pnode,
#         loss_manager=loss_manager,
#         ode_solver=odeint,
#         method=cfg.method,
#         rtol=cfg.rtol,
#         atol=cfg.atol,
#         device=cfg.device,
#     ).to(cfg.device)

#     return model

def sample_candidate_interior(train_dataset, n, device):
    x_phys = train_dataset._sample_states_phys(n)
    x_net = train_dataset._scale_states(x_phys)

    tau_phys = train_dataset._sample_tau_phys(n)
    tau_net = train_dataset._scale_tau(tau_phys)

    xt = torch.cat([x_net, tau_net], dim=-1)

    return {
        "xt": xt.to(device).float().requires_grad_(True),
        "x_phys": x_phys.to(device).float(),
        "tau_phys": tau_phys.squeeze(-1).to(device).float(),
    }


def residual_weighted_pool(model, residual, train_dataset, device, num_candidates, k=1.0, c=1.0):
    cand = sample_candidate_interior(train_dataset, num_candidates, device)

    xt = cand["xt"]
    tau_phys = cand["tau_phys"]

    V, raw, latent, dlatent = model.compute_value_at_xt(xt, tau_phys)
    r = residual.compute_residual(
        model=model,
        V=V,
        raw=raw,
        xt=xt,
        latent=latent,
        dlatent=dlatent,
    )

    scores = r.abs().detach()
    weights = scores.pow(k)
    weights = weights / (weights.mean() + c)
    probs = weights / weights.sum()

    return {
        "xt": cand["xt"].detach().cpu(),
        "x_phys": cand["x_phys"].detach().cpu(),
        "tau_phys": cand["tau_phys"].detach().cpu(),
        "scores": scores.detach().cpu(),
        "probs": probs.detach().cpu(),
    }


def draw_adaptive_points(pool, num_draw):
    idx = torch.multinomial(pool["probs"], num_samples=num_draw, replacement=True)
    return {
        "xt": pool["xt"][idx],
        "x_phys": pool["x_phys"][idx],
        "tau_phys": pool["tau_phys"][idx],
    }


def inject_adaptive_points(batch, adaptive_pts, hard_frac=0.25):
    n_total = batch["xt_interior"].shape[0]
    n_hard = int(hard_frac * n_total)
    n_uniform = n_total - n_hard

    batch["xt_interior"] = torch.cat(
        [batch["xt_interior"][:n_uniform], adaptive_pts["xt"][:n_hard]], dim=0
    )
    batch["x_interior_phys"] = torch.cat(
        [batch["x_interior_phys"][:n_uniform], adaptive_pts["x_phys"][:n_hard]], dim=0
    )
    batch["tau_interior_phys"] = torch.cat(
        [batch["tau_interior_phys"][:n_uniform], adaptive_pts["tau_phys"][:n_hard]], dim=0
    )
    return batch

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

    pnode = PNODE(
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
    # last = model.pnode.model.model[-1]
    # torch.nn.init.zeros_(last.weight)
    # torch.nn.init.zeros_(last.bias)
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

            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.pnode.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([model.alpha0], max_norm=1.0)

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

    # -------------------------
    # Full-horizon CNF-ROM-style training
    # -------------------------
    model.loss_manager.loss_weights = {"terminal": 1.0, "pinn": 1.0}
    
    # sample tau over the whole horizon from the start
    train_dataset.set_tau_max(cfg.T)
    
    # keep total training budget about the same as before
    if hasattr(cfg, "total_steps"):
        total_steps = cfg.total_steps
    else:
        total_steps = cfg.steps_per_stage * len(cfg.tau_schedule)
    
    if hasattr(cfg, "save_freq"):
        save_freq = cfg.save_freq
    else:
        save_freq = cfg.steps_per_stage
    
    step = 0
    adaptive_pool = None

    for step in range(total_steps):
        batch = next(iter(train_loader))
    
        if cfg.adaptive_sampling:
            n_hard = int(cfg.hard_frac * batch["xt_interior"].shape[0])
    
            if adaptive_pool is None or step % cfg.adaptive_every == 0:
                adaptive_pool = residual_weighted_pool(
                    model=model,
                    residual=residual,
                    train_dataset=train_dataset,
                    device=cfg.device,
                    num_candidates=cfg.candidate_mult * n_hard,
                    k=cfg.rad_k,
                    c=cfg.rad_c,
                )
    
            adaptive_pts = draw_adaptive_points(adaptive_pool, n_hard)
            batch = inject_adaptive_points(batch, adaptive_pts, hard_frac=cfg.hard_frac)
    
        for opt in optimizers.values():
            opt.zero_grad()
    
        results = model.compute_losses(batch, mode="train")
        results["loss_train"].backward()
    
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.pnode.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([model.alpha0], max_norm=1.0)
    
        for opt in optimizers.values():
            opt.step()
    
        if (step+1) % save_freq == 0 or (step+1) == total_steps:
            print(f"step={step}/{total_steps}")

            save_checkpoint(
                ckpt_dir / f"step_{step:06d}.pt",
                model,
                optimizers,
                step=step,
                tau_max=cfg.T,
                cfg=cfg,
            )

            # optional: make plots at all evaluation taus
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
