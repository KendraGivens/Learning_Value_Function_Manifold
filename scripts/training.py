import os, json, argparse, yaml, torch, lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from lvfm.data_generation import generate_burgers_solution_grid
from lvfm.datasets import BurgersExactDataset, BurgersPhysicsDataset
from lvfm.models import Decoder, PNODE, PNODEFunc, CNFROM, ModelWrapper
from lvfm.plotting import save_loss_plot, plot_heatmap_compare, plot_heatmap_physics

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config_copy(cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

def make_data_loader(cfg):
    mu_train = cfg["mu_train"]
    mu_test = cfg.get("mu_test", [])
    mu_all = sorted(list(mu_train) + list(mu_test))
    
    n_x = int(cfg["n_x"])
    n_t = int(cfg["n_t"])
    T_f = float(cfg["T_f"])
    
    n_data_samples = int(cfg["n_data_samples"])
    n_physics_samples = int(cfg["n_physics_samples"])
    data_batch_size = int(cfg["data_batch_size"])
    physics_batch_size = int(cfg["physics_batch_size"])

    x_axis, t_axis, mu_axis, u_grid = generate_burgers_solution_grid(mu_train, n_x, n_t, T_f)

    exact_train_dataset = BurgersExactDataset(x_axis, t_axis, mu_axis, u_grid, n_samples=n_data_samples)
    exact_train_loader  = DataLoader(exact_train_dataset, batch_size=data_batch_size, shuffle=True)
    
    physics_train_dataset = BurgersPhysicsDataset(mu_all, n_samples=n_physics_samples)
    physics_train_loader  = DataLoader(physics_train_dataset, batch_size=physics_batch_size, shuffle=True)

    return exact_train_loader, physics_train_loader

def make_trainer(ckpt_dir, log_root, stage_name, max_epochs, log_every_n_steps, accelerator, devices, monitor):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_ckpt_cb = ModelCheckpoint(dirpath=str(ckpt_dir), filename="best-{epoch:04d}", monitor=monitor, mode="min", save_last=True, save_top_k=1, auto_insert_metric_name=False)
    periodic_ckpt_cb = ModelCheckpoint(dirpath=str(ckpt_dir), filename="epoch-{epoch:04d}", every_n_epochs=10, save_top_k=-1, auto_insert_metric_name=False)

    cbs = [best_ckpt_cb, periodic_ckpt_cb, LearningRateMonitor(logging_interval="epoch"), RichProgressBar()]

    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(str(log_root), name=stage_name, version=0)

    trainer = L.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=devices, deterministic=True, log_every_n_steps=log_every_n_steps, callbacks=cbs, logger=logger)

    return trainer, logger, cbs

def fit_with_resume(trainer, model, loader, ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    last_ckpt = ckpt_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if last_ckpt.exists() else None
    trainer.fit(model, loader, ckpt_path=ckpt_path)

def load_pretrained_wrapper(latent_dim, pretrained_ckpt, device):
    decoder = Decoder(latent_dim=latent_dim)
    pnode_func = PNODEFunc(latent_dim=latent_dim)
    pnode = PNODE(pnode_func, latent_dim=latent_dim)
    cnf = CNFROM(decoder, pnode)
    wrapper = ModelWrapper(cnf)
    ckpt = torch.load(str(pretrained_ckpt), map_location="cpu")
    wrapper.load_state_dict(ckpt["state_dict"], strict=False)
    wrapper.to(device).eval()
    return wrapper

def main():
    p = argparse.ArgumentParser("Train CNFROM")
    p.add_argument("--config", required=True)
    p.add_argument("--devices", default=None)         
    args = p.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "configs" / f"{args.config}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_yaml(config_path)
    seed = int(config["seed"])
    task = str(config["task"])

    config_name = Path(args.config).stem
    run_name = f"{config_name}_seed_{seed}" if config_name == task else f"{config_name}_{task}_seed_{seed}"

    ckpt_root = Path(config["ckpt_root"])
    run_ckpt_dir = ckpt_root / run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(config, run_ckpt_dir)

    log_root = Path("logs") / run_name
    log_root.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = int(args.devices) if args.devices is not None else int(config["device"])

    log_every_n_steps = int(config["log_every_n_steps"])
    latent_dim = int(config["latent_dim"])
    max_epochs_data = int(config["max_epochs_data"])
    max_epochs_physics = int(config["max_epochs_physics"])
    lr = float(config.get("lr", 1e-3))
    
    L.seed_everything(seed, workers=True)
    
    exact_train_loader, physics_train_loader = make_data_loader(config)

    decoder = Decoder(latent_dim=latent_dim)
    pnode_func = PNODEFunc(latent_dim=latent_dim)
    pnode = PNODE(pnode_func, latent_dim=latent_dim)
    cnf = CNFROM(decoder, pnode)
    model = ModelWrapper(cnf, lr=lr)

    data_ckpt_dir = run_ckpt_dir / "data"
    physics_ckpt_dir = run_ckpt_dir / "physics"
    pretrained_ckpt = data_ckpt_dir / "pretrained.ckpt"

    if task in ("dual_frozen", "dual_unfrozen"):
        model.mode = "data"
        data_trainer, data_logger, _ = make_trainer(data_ckpt_dir, log_root, "data", max_epochs_data, log_every_n_steps, accelerator, devices, "train_loss_data")
        fit_with_resume(data_trainer, model, exact_train_loader, data_ckpt_dir)

        data_trainer.save_checkpoint(str(pretrained_ckpt))

        if task == "dual_frozen":
            for p in model.model.decoder.parameters():
                p.requires_grad = False

    model.mode = "physics"
    physics_trainer, physics_logger, _ = make_trainer(physics_ckpt_dir, log_root, "physics", max_epochs_physics, log_every_n_steps, accelerator, devices, "train_loss_physics")
    fit_with_resume(physics_trainer, model, physics_train_loader, physics_ckpt_dir)

    plot_dir = log_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_loss_plot(model, plot_dir)

    mu0 = float(config.get("mu_plot", config["mu_train"][0]))
    if pretrained_ckpt.exists():
        pre_wrapper = load_pretrained_wrapper(latent_dim, pretrained_ckpt, model.device)
        plot_heatmap_compare(pre_wrapper.model, model.model, plot_dir, mu0=mu0)
    else:
        plot_heatmap_physics(model.model, plot_dir, mu0=mu0)

if __name__ == "__main__":
    main()
