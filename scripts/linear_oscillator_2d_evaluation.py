import os 
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from torchdiffeq import odeint
from lvfm.datasets import LinearOscillator2DDataset
from lvfm.residuals import LinearOscillator2DResidual
from lvfm.managers import LossManager
from lvfm.models import Decoder, PNODE, INR_PNODE
from lvfm.hj_solvers import solve_linear_oscillator_2d
from lvfm.helpers import compute_metrics

def to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**d)
    return d
    
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_residual(cfg):
    return LinearOscillator2DResidual(
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

def scale_states_to_net(x_phys, x1_bounds, x2_bounds, scale_to_minus1_1=True):
    if not scale_to_minus1_1:
        return x_phys
    bounds = torch.tensor([x1_bounds, x2_bounds], dtype=x_phys.dtype, device=x_phys.device)
    low = bounds[:, 0]
    high = bounds[:, 1]
    return 2.0 * (x_phys - low) / (high - low) - 1.0 

def load_cfg_and_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path)
    if "cfg" in ckpt and ckpt["cfg"] is not None:
        cfg = to_namespace(ckpt["cfg"])

    cfg.device = device
    residual = create_residual(cfg)
    model = create_model(cfg, residual)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return cfg, model

def sample_batch(cfg, num_interior, num_terminal, tau_max):
    dataset = LinearOscillator2DDataset(
        num_batches=1,
        num_interior=num_interior,
        num_terminal=num_terminal,
        T=cfg.T,
        x1_bounds=cfg.x1_bounds,
        x2_bounds=cfg.x2_bounds,
        tau_max=tau_max,
        scale_to_minus1_1=cfg.scale_to_minus1_1,
        scale_time_to_0_1=cfg.scale_time_to_01,
    )
    return dataset[0]
    
@torch.no_grad()
def eval_model_on_grid(model, cfg, tau_phys, nx=201, chunk_size=4096):
    x1 = torch.linspace(cfg.x1_bounds[0], cfg.x1_bounds[1], nx, device=cfg.device)
    x2 = torch.linspace(cfg.x2_bounds[0], cfg.x2_bounds[1], nx, device=cfg.device)

    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    x_phys = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1)
    x_net = scale_states_to_net(x_phys, cfg.x1_bounds, cfg.x2_bounds, cfg.scale_to_minus1_1)

    tau_vec = torch.full((x_phys.shape[0],), float(tau_phys), dtype=torch.float32, device=cfg.device)
    tau_net = tau_vec[:, None] / cfg.T if cfg.scale_time_to_01 else tau_vec[:, None]

    preds = []
    for start in range(0, x_phys.shape[0], chunk_size):
        end = min(start + chunk_size, x_phys.shape[0])
        xt = torch.cat([x_net[start:end], tau_net[start:end]], dim=-1).float()
        V, _, _ = model.compute_value_at_xt(xt, tau_vec[start:end])
        preds.append(V.detach().reshape(-1).cpu())

    return torch.cat(preds, dim=0).numpy().reshape(nx, nx)

def compute_losses(model, cfg, num_interior, num_terminal, num_batches, eval_tau):
    model.eval()
    pinn_losses = []
    term_losses = []

    for _ in range(num_batches):
        batch = sample_batch(cfg, num_interior, num_terminal, eval_tau)

        xt_interior = batch["xt_interior"].to(cfg.device).float().requires_grad_(True)
        tau_interior_phys = batch["tau_interior_phys"].to(cfg.device).float()
        xt_terminal = batch["xt_terminal"].to(cfg.device).float()
        x_terminal_phys = batch["x_terminal_phys"].to(cfg.device).float()

        V, latent, dlatent = model.compute_value_at_xt(xt_interior, tau_interior_phys)
        V_terminal = model.decode_terminal(xt_terminal)

        loss_dict = model.loss_manager.compute_losses(model, xt_interior, x_terminal_phys, V, V_terminal, latent, dlatent)

        pinn_losses.append(loss_dict["loss_pinn"].detach().item())
        term_losses.append(loss_dict["loss_terminal"].detach().item())

    return {
        "pinn_loss": float(np.mean(pinn_losses)),
        "terminal_loss": float(np.mean(term_losses))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--num_interior", type=int, default=8192)
    parser.add_argument("--num_terminal", type=int, default=8192)
    parser.add_argument("--num_eval_batches", type=int, default=8)
    args = parser.parse_args()

    device_idx = 0 if args.device is None else args.device
    device = torch.device(f"cuda:{device_idx}")
    
    runs_dir = Path("runs/linear_oscillator_2d")
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "ckpts" / "tau_1.00.pt").exists()])
    rows = []

    eval_tau = 1.0
    
    for run_dir in run_dirs:
        checkpoint_dir = run_dir / "ckpts"
        checkpoint_path = checkpoint_dir / "tau_1.00.pt"

        cfg, model = load_cfg_and_model(checkpoint_path, device)

        tau_schedule = [float(t) for t in cfg.tau_schedule]
        ground_truth_value_function = solve_linear_oscillator_2d(cfg.tau_schedule, args.nx, args.nx, cfg.x1_bounds, cfg.x2_bounds, cfg.u_bound, cfg.d_bound, cfg.omega, cfg.beta)
        tau_index = tau_schedule.index(eval_tau)
        ground_truth_brt = np.array(ground_truth_value_function[tau_index+1]).T

        pred_brt = eval_model_on_grid(model, cfg, eval_tau, args.nx, args.chunk_size)
        losses = compute_losses(model, cfg, args.num_interior, args.num_terminal, args.num_eval_batches, eval_tau)
        mae, overlap = compute_metrics(pred_brt, ground_truth_brt)

        row = {
            "run_name": run_dir.name,
            "mae": round(mae, 4),
            "overlap": round(overlap, 2),
            "pinn_loss": round(losses["pinn_loss"], 4),
            "terminal_loss": round(losses["terminal_loss"], 4),
        }
        rows.append(row)

        print(f"{run_dir.name}")

    out_csv = runs_dir / "eval_tau_1.00.csv"
    
    if rows:
        best_mae = min(rows, key=lambda r: r["mae"])
        best_overlap = max(rows, key=lambda r: r["overlap"])
        best_pinn = min(rows, key=lambda r: r["pinn_loss"])
        best_terminal = min(rows, key=lambda r: r["terminal_loss"])
    
        blank_row = {k: "" for k in rows[0].keys()}
    
        summary_rows = [
            {
                "run_name": f"__BEST_MAE__: {best_mae['run_name']}",
                "mae": round(best_mae["mae"], 4),
                "overlap": "",
                "pinn_loss": "",
                "terminal_loss": "",
            },
            {
                "run_name": f"__BEST_OVERLAP__: {best_overlap['run_name']}",
                "mae": "",
                "overlap": round(best_overlap["overlap"], 2),
                "pinn_loss": "",
                "terminal_loss": "",
            },
            {
                "run_name": f"__BEST_PINN__: {best_pinn['run_name']}",
                "mae": "",
                "overlap": "",
                "pinn_loss": round(best_pinn["pinn_loss"], 4),
                "terminal_loss": "",
            },
            {
                "run_name": f"__BEST_TERMINAL__: {best_terminal['run_name']}",
                "mae": "",
                "overlap": "",
                "pinn_loss": "",
                "terminal_loss": round(best_terminal["terminal_loss"], 4),
            },
        ]
    
        rows_to_write = rows + [blank_row] + summary_rows
    
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows_to_write)


if __name__ == "__main__":
    main()
        




















    