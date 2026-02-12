import os, json, argparse, yaml, torch, lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from lvfm.datasets import Air3DPhysicsDataset
from lvfm.models import Air3DDecoder, Air3DPNODE, Air3DPNODEFunc, Air3DCNFROM, Air3DModelWrapper
from lvfm.plotting import plot_Air3D_loss, plot_Air3D_heatmap
from lvfm.callbacks import PlotEveryNEpochs

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
    n_samples = int(cfg["n_samples"])
    batch_size = int(cfg["batch_size"])

    train_dataset = Air3DPhysicsDataset(n_samples=n_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataset, train_loader

def make_trainer(ckpt_dir, log_root, stage_name, max_epochs, log_every_n_steps, accelerator, devices, extra_callbacks=None):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_ckpt_cb = ModelCheckpoint(dirpath=str(ckpt_dir), filename="best-{epoch:04d}", monitor="train_loss", mode="min", save_last=True, save_top_k=1, auto_insert_metric_name=False)
    periodic_ckpt_cb = ModelCheckpoint(dirpath=str(ckpt_dir), filename="epoch-{epoch:04d}", every_n_epochs=100, save_top_k=-1, auto_insert_metric_name=False)

    cbs = [best_ckpt_cb, periodic_ckpt_cb, LearningRateMonitor(logging_interval="epoch"), RichProgressBar()]

    if extra_callbacks:
        cbs.extend(extra_callbacks)
        
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
    decoder = Air3DDecoder(latent_dim=latent_dim)
    pnode_func = Air3DPNODEFunc(latent_dim=latent_dim)
    pnode = Air3DPNODE(pnode_func, latent_dim=latent_dim)
    cnf = Air3DCNFROM(decoder, pnode)
    wrapper = Air3DModelWrapper(cnf)
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

    config_name = Path(args.config).stem
    run_name = f"{config_name}_seed_{seed}"
    
    ckpt_root = Path(config["ckpt_root"])
    run_ckpt_dir = ckpt_root / run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(config, run_ckpt_dir)

    log_root = Path("logs") / run_name
    log_root.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = int(args.devices) if args.devices is not None else int(config["device"])

    plot_every_n_epochs = int(config["plot_every_n_epochs"])
    log_every_n_steps = int(config["log_every_n_steps"])
    latent_dim = int(config["latent_dim"])
    max_epochs = int(config["max_epochs"])
    lr = float(config.get("lr", 1e-3))
    
    L.seed_everything(seed, workers=True)
    
    train_dataset, train_loader = make_data_loader(config)

    decoder = Air3DDecoder(latent_dim=latent_dim)
    pnode_func = Air3DPNODEFunc(latent_dim=latent_dim)
    pnode = Air3DPNODE(pnode_func, latent_dim=latent_dim)
    cnf = Air3DCNFROM(decoder, pnode)
    model = Air3DModelWrapper(cnf, train_dataset, lr=lr)

    ckpt_dir = run_ckpt_dir
    plot_dir = log_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_cb = PlotEveryNEpochs(plot_dir=plot_dir, every_n_epochs=plot_every_n_epochs)
    
    trainer, logger, _ = make_trainer(ckpt_dir=ckpt_dir, log_root=log_root, stage_name="physics", max_epochs=max_epochs, log_every_n_steps=log_every_n_steps, accelerator=accelerator, devices=devices, extra_callbacks=[plot_cb])
    fit_with_resume(trainer, model, train_loader, ckpt_dir)

    plot_dir = log_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
