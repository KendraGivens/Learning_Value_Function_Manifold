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
from lvfm.plotting import plot_linear_oscillator_2d

def to_namespace(d):
    return SimpleNamespace(**d)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_dataloader(cfg):    
    train_dataset = LinearOscillator2DDataset(
        num_batches=cfg.num_batches,
        num_interior=cfg.num_interior,
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
    
    cfg_path = Path("configs") / f"{args.cfg}.yaml"
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
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, run_dir /"config.yaml")
            
    train_dataset, train_loader = create_dataloader(cfg)
    residual = create_residual(cfg)
    model = create_model(cfg, residual)
    optimizers = model.create_optimizers(lr=cfg.lr)

    # pretraining
    train_dataset.set_tau_max(0.0)
    model.loss_manager.loss_weights = {"terminal": 1.0, "pinn": 0.0}
    # pbar = trange(cfg.pretrain_steps, desc="pretrain tau=0.00")
    step = 0
    print("training")
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

            # if step % 50 == 0:
            #     print(
            #         f"pretrain step {step:4d} | "
            #         f"loss={results['loss_train'].item():.3e} | "
            #         f"term={results['loss_terminal'].item():.3e}"
            #     )

            # pbar.set_postfix(
            #     loss=f"{results['loss_train'].item():.2e}",
            #     term=f"{results['loss_terminal'].item():.2e}",
            #     pinn=f"{results['loss_pinn'].item():.2e}",
            #     )
            # pbar.update(1)
            step += 1
            if step >= cfg.pretrain_steps:
                break
    # pbar.close()
    plot_linear_oscillator_2d(
        model,
        device=cfg.device,
        tau=0.0,
        x1_bounds=cfg.x1_bounds,
        x2_bounds=cfg.x2_bounds,
        nx=201,
        chunk_size=4096,
        scale_to_minus1_1=cfg.scale_to_minus1_1,
        T=cfg.T,
        scale_time_to_01=cfg.scale_time_to_01,
        threshold=cfg.beta,
        show_terminal_target=True,
        title="Terminal pretraining (tau=0)",
        save_path=plot_dir / "pretrain_tau_0.00.png",
    )
    print("finished pretraining")
    # curriculum training
    model.loss_manager.loss_weights = {"terminal": 1.0, "pinn": 1.0}
    for tau_max in cfg.tau_schedule:
        train_dataset.set_tau_max(tau_max)
        print(f"\nTraining with tau_max = {tau_max:.2f}")
        # pbar = trange(cfg.steps_per_stage, desc=f"tau={tau_max:.2f}")
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

                # if step % 50 == 0:
                #     print(
                #         f"step {step:4d} | "
                #         f"loss={results['loss_train'].item():.3e} | "
                #         f"term={results['loss_terminal'].item():.3e} | "
                #         f"pinn={results['loss_pinn'].item():.3e}"
                #     )
                # pbar.set_postfix(
                #     loss=f"{results['loss_train'].item():.2e}",
                #     term=f"{results['loss_terminal'].item():.2e}",
                #     pinn=f"{results['loss_pinn'].item():.2e}",
                #     )
                # pbar.update(1)
                step += 1
                if step >= cfg.steps_per_stage:
                    break
        plot_linear_oscillator_2d(
            model,
            device=cfg.device,
            tau=tau_max,
            x1_bounds=cfg.x1_bounds,
            x2_bounds=cfg.x2_bounds,
            nx=201,
            chunk_size=4096,
            scale_to_minus1_1=cfg.scale_to_minus1_1,
            T=cfg.T,
            scale_time_to_01=cfg.scale_time_to_01,
            threshold=cfg.beta,
            show_terminal_target=(tau_max == 0.0),
            title=f"Learned boundary at tau={tau_max:.2f}",
            save_path=plot_dir / f"tau_{tau_max:.2f}.png",
        )
    # pbar.close()
    print("finished curriculum")
    
if __name__ == "__main__":
    main()
