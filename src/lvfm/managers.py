import torch
from lvfm.training import causal_temporal_loss

class LossManager:
    def __init__(self, residual, loss_weights=None):
        self.residual = residual
        self.loss_weights = {
            "terminal": 0.0,
            "pinn": 1.0,
        } if loss_weights is None else loss_weights

    def compute_losses(
        self,
        model,
        xt_interior,
        V=None,
        raw=None,
        latent=None,
        dlatent=None,
        raw_terminal=None,
        deepreach=False,
        causal_loss=False,
        causal_chunks=16,
        causal_eps=1.0,
    ):
        if deepreach:
            residual = self.residual.compute_deepreach_residual(
                model=model,
                xt=xt_interior,
            )
        else:
            residual = self.residual.compute_cnf_rom_residual(
                model=model,
                V=V,
                raw=raw,
                xt=xt_interior,
                latent=latent,
                dlatent=dlatent,
            )

        if causal_loss:
            tau_phys = self.residual._tau_phys_from_xt(xt_interior)

            loss_pinn, causal_weights, causal_chunk_losses = causal_temporal_loss(
                residual=residual,
                tau_phys=tau_phys,
                T=self.residual.T,
                num_chunks=causal_chunks,
                eps=causal_eps,
                normalize_losses=True,
            )
        else:
            loss_pinn = residual.abs().mean()
            causal_weights = None
            causal_chunk_losses = None

        if raw_terminal is not None and self.loss_weights.get("terminal", 0.0) > 0.0:
            loss_terminal = raw_terminal.squeeze(-1).abs().mean()
        else:
            loss_terminal = torch.zeros((), device=xt_interior.device)

        loss_train = (
            self.loss_weights.get("terminal", 0.0) * loss_terminal
            + self.loss_weights.get("pinn", 1.0) * loss_pinn
        )

        out = {
            "loss_terminal": loss_terminal,
            "loss_pinn": loss_pinn,
            "loss_train": loss_train,
        }

        if causal_weights is not None:
            out["causal_weights"] = causal_weights.detach()
            out["causal_chunk_losses"] = causal_chunk_losses.detach()

        return out