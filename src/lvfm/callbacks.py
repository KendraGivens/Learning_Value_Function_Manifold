from lightning.pytorch.callbacks import Callback
from pathlib import Path
from lvfm.plotting import plot_Air3D_loss, plot_Air3D_heatmap

class PlotEveryNEpochs(Callback):
    def __init__(self, plot_dir: Path, every_n_epochs: int = 25, heatmap_tau: float = 1.0):
        super().__init__()
        self.plot_dir = Path(plot_dir)
        self.every_n_epochs = int(every_n_epochs)
        self.heatmap_tau = float(heatmap_tau)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # 1-based for filenames
        if epoch % self.every_n_epochs != 0 and epoch != trainer.max_epochs:
            return

        self.plot_dir.mkdir(parents=True, exist_ok=True)

        plot_Air3D_loss(pl_module, self.plot_dir)

        epoch_dir = self.plot_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        plot_Air3D_heatmap(pl_module, epoch_dir) 
 