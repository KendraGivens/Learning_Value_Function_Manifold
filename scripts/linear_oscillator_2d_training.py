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
from lvfm.models import Decoder, PNODE, INR_PNODE
from lvfm.hj_solvers import solve_linear_oscillator_2d 
from lvfm.plotting import plot_linear_oscillator_2d

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
    
    pnode = PNODE(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.pnode_hidden_dim,
    )
    
    model = INR_PNODE(
        latent_dim=cfg.latent_dim,
        decoder=decoder,
        pnode=pnode,
        loss_manager=loss_manager,
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
    
    cfg.device = torch.device("cuda")

    run_name = args.cfg
    run_dir = Path("runs") / run_name
    ckpt_dir = run_dir / "checkpoints"
    plot_dir = run_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, run_dir / "config.yaml")
            
    train_dataset, train_loader = create_dataloader(cfg)
    residual = create_residual(cfg)
    model = create_model(cfg, residual)
    optimizers = model.create_optimizers(lr=cfg.lr)

    ground_truth_value_function = solve_linear_oscillator_2d(cfg.tau_schedule, 201, 201, cfg.x1_bounds, cfg.x2_bounds, cfg.u_bound, cfg.d_bound, cfg.omega, cfg.beta)
    
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
    gt_slice = np.array(ground_truth_value_function[0])
    plot_linear_oscillator_2d(
        model,
        device=cfg.device,
        tau=0.0,
        gt_values=gt_slice,
        x1_bounds=cfg.x1_bounds,
        x2_bounds=cfg.x2_bounds,
        nx=201,
        chunk_size=4096,
        scale_to_minus1_1=cfg.scale_to_minus1_1,
        T=cfg.T,
        scale_time_to_01=cfg.scale_time_to_01,
        title="Predicted vs Ground Truth at tau=0.00",
        save_path=plot_dir / "compare_tau_0.00.png",
    )
    
    # curriculum training
    model.loss_manager.loss_weights = {"terminal": 1.0, "pinn": 1.0}
    for i, tau_max in enumerate(cfg.tau_schedule):
        train_dataset.set_tau_max(tau_max)
        print(f"\nTraining with tau_max = {tau_max:.2f}")
        step = 0
        while step < cfg.steps_per_stage:
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
                if step >= cfg.steps_per_stage:
                    break
        save_checkpoint(ckpt_dir/f"tau_{tau_max:.2f}.pt", model, optimizers, step=step, tau_max=tau_max, cfg=cfg)  
        
        gt_slice = np.array(ground_truth_value_function[i + 1])
        plot_linear_oscillator_2d(
            model,
            device=cfg.device,
            tau=tau_max,
            gt_values=gt_slice,
            x1_bounds=cfg.x1_bounds,
            x2_bounds=cfg.x2_bounds,
            nx=201,
            chunk_size=4096,
            scale_to_minus1_1=cfg.scale_to_minus1_1,
            T=cfg.T,
            scale_time_to_01=cfg.scale_time_to_01,
            title=f"Predicted vs Ground Truth at tau={tau_max:.2f}",
            save_path=plot_dir / f"compare_tau_{tau_max:.2f}.png",
        )
if __name__ == "__main__":
    main()
