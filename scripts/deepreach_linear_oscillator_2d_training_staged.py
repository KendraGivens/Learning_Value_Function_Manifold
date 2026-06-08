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
from lvfm.datasets import LinearOscillator2DDataset
from lvfm.residuals import LinearOscillator2DResidual
from lvfm.managers import LossManager
from lvfm.models import DeepReachModel, DeepReachExact
from lvfm.hj_solvers import solve_linear_oscillator_2d 
from lvfm.plotting import plot_linear_oscillator_2d_deepreach

def to_namespace(d):
    return SimpleNamespace(**d)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_checkpoint(path, model, optimizer, step=None, tau_max=None, cfg=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tau_max": tau_max
    }
    if cfg is not None: 
        ckpt["cfg"] = vars(cfg) if hasattr(cfg, "__dict__") else cfg

    torch.save(ckpt, path)

def create_dataloader(cfg):    
    train_dataset = LinearOscillator2DDataset(
        num_batches=cfg.num_batches,
        num_interior= cfg.num_interior,
        num_terminal=cfg.num_terminal,
        T=cfg.T,
        x1_bounds=cfg.x1_bounds,
        x2_bounds=cfg.x2_bounds,
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
    residual = LinearOscillator2DResidual(
            oscillation_speed=cfg.omega,
            control_bound=cfg.u_bound,
            disturbance_bound=cfg.d_bound,
            radius=cfg.beta,
            T=cfg.T,
            x1_bounds=cfg.x1_bounds,
            x2_bounds=cfg.x2_bounds,
            scale_to_minus1_1=cfg.scale_to_minus1_1,
            scale_time_to_01=cfg.scale_time_to_01
    )
    return residual

def create_model(cfg, residual):
    loss_manager = LossManager(
        residual=residual,
        loss_weights={"terminal": 1.0, "pinn": 1.0},
    )

    raw_model = DeepReachModel(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        omega0=cfg.omega0,
    ).to(cfg.device)

    model = DeepReachExact(
        backbone=raw_model,
        residual=residual,
        coordinate_dim=2,  
        value_var=0.5,
        value_normto=0.02,
    ).to(cfg.device)

    return model, loss_manager

def compute_losses(model, loss_manager, batch, device):
    xt_interior = batch["xt_interior"].to(device).float().requires_grad_(True)
    xt_terminal = batch["xt_terminal"].to(device).float()
    x_terminal_phys = batch["x_terminal_phys"].to(device).float()

    loss_dict = loss_manager.compute_losses(
        model=model,
        xt_interior=xt_interior,
        xt_terminal=xt_terminal,
        x_terminal_phys=x_terminal_phys,
        V=None,
        V_terminal=None,
        latent=None,
        dlatent=None,
        deepreach=True,
    )

    loss_train = (
        loss_manager.loss_weights["terminal"] * loss_dict["loss_terminal"]
        + loss_manager.loss_weights["pinn"] * loss_dict["loss_pinn"]
    )
    loss_dict["loss_train"] = loss_train
    return loss_dict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--device", type=int)
    args = p.parse_args()
    
    cfg_path = Path("configs") / "linear_oscillator_2d" / f"{args.cfg}.yaml"
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
    run_dir = Path("runs/linear_oscillator_2d") / run_name
    ckpt_dir = run_dir / "ckpts"
    plot_dir = run_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, run_dir / "config.yaml")
            
    train_dataset, train_loader = create_dataloader(cfg)
    residual = create_residual(cfg)
    model, loss_manager = create_model(cfg, residual)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ground_truth_value_function = solve_linear_oscillator_2d(cfg.tau_schedule, 201, 201, cfg.x1_bounds, cfg.x2_bounds, cfg.u_bound, cfg.d_bound, cfg.omega, cfg.beta)
    
    # pretraining
    train_dataset.set_tau_max(0.0)
    loss_manager.loss_weights = {"terminal": 1.0, "pinn": 0.0}
    step = 0
    while step < cfg.pretrain_steps:
        for batch in train_loader:
            optimizer.zero_grad()

            results = compute_losses(model, loss_manager, batch, cfg.device)
            results["loss_train"].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            step += 1
            if step >= cfg.pretrain_steps:
                break

    save_checkpoint(ckpt_dir/"pretrain.pt", model, optimizer, step=step, tau_max=0.0, cfg=cfg)
    gt_slice = np.array(ground_truth_value_function[0]).T
    plot_linear_oscillator_2d_deepreach(model, device=cfg.device, tau=0.0, gt_values=gt_slice, x1_bounds=cfg.x1_bounds, x2_bounds=cfg.x2_bounds, nx=201, chunk_size=4096, scale_to_minus1_1=cfg.scale_to_minus1_1, T=cfg.T, scale_time_to_01=cfg.scale_time_to_01, title="Predicted vs Ground Truth at tau=0.00", save_path=plot_dir / "compare_tau_0.00.png")
    
    # curriculum training
    loss_manager.loss_weights = {"terminal": 1.0, "pinn": 1.0}
    for i, tau_max in enumerate(cfg.tau_schedule):
        train_dataset.set_tau_max(tau_max)
        print(f"\nTraining with tau_max = {tau_max:.2f}")
        step = 0
        while step < cfg.steps_per_stage:
            for batch in train_loader:
                optimizer.zero_grad()

                results = compute_losses(model, loss_manager, batch, cfg.device)
                results["loss_train"].backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                    
                step += 1
                if step >= cfg.steps_per_stage:
                    break
        save_checkpoint(ckpt_dir/f"tau_{tau_max:.2f}.pt", model, optimizer, step=step, tau_max=tau_max, cfg=cfg)  
        
        gt_slice = np.array(ground_truth_value_function[i + 1]).T
        plot_linear_oscillator_2d_deepreach(model, device=cfg.device, tau=tau_max, gt_values=gt_slice, x1_bounds=cfg.x1_bounds, x2_bounds=cfg.x2_bounds, nx=201, chunk_size=4096, scale_to_minus1_1=cfg.scale_to_minus1_1, T=cfg.T, scale_time_to_01=cfg.scale_time_to_01, title=f"Predicted vs Ground Truth at tau={tau_max:.2f}", save_path=plot_dir / f"compare_tau_{tau_max:.2f}.png")

if __name__ == "__main__":
    main()
