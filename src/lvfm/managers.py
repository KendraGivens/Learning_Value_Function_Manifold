import torch.nn as nn

class LossManager():
    def __init__(self, residual, loss_weights=None):
        self.residual = residual
        self.loss_weights = {"terminal": 1.0, "pinn": 1.0} if loss_weights is None else loss_weights
        self.criterion = nn.MSELoss()
        
    def compute_losses(self, model, xt_interior, x_terminal_phys, V, V_terminal, latent, dlatent):
        target_level_set = self.residual.target_function(x_terminal_phys)
    
        loss_terminal = self.criterion(V_terminal, target_level_set)
        loss_pinn, _ = self.residual.compute_loss(
            model=model,
            V=V,
            xt=xt_interior,
            latent=latent,
            dlatent=dlatent,
        )
    
        return {
            "loss_terminal": loss_terminal,
            "loss_pinn": loss_pinn,
        }