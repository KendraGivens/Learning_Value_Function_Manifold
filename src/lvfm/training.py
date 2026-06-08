import torch 

def causal_temporal_loss(
    residual,
    tau_phys,
    T,
    num_chunks=16,
    eps=1.0,
    normalize_losses=True,
):

    residual = residual.reshape(-1)
    tau_phys = tau_phys.reshape(-1).detach()

    point_losses = residual.abs()

    edges = torch.linspace(
        0.0,
        float(T),
        steps=num_chunks + 1,
        device=residual.device,
        dtype=tau_phys.dtype,
    )

    chunk_losses = []

    for i in range(num_chunks):
        if i == num_chunks - 1:
            mask = (tau_phys >= edges[i]) & (tau_phys <= edges[i + 1])
        else:
            mask = (tau_phys >= edges[i]) & (tau_phys < edges[i + 1])

        if mask.any():
            chunk_losses.append(point_losses[mask].mean())
        else:
            # Empty chunk contributes no direct loss.
            chunk_losses.append(torch.zeros((), device=residual.device))

    chunk_losses = torch.stack(chunk_losses)

    # Build weights without backpropagating through the weights.
    with torch.no_grad():
        detached = chunk_losses.detach()

        if normalize_losses:
            detached = detached / detached.mean().clamp_min(1e-8)

        cumulative_previous = torch.cat(
            [
                torch.zeros(1, device=residual.device),
                torch.cumsum(detached, dim=0)[:-1],
            ],
            dim=0,
        )

        weights = torch.exp(-float(eps) * cumulative_previous)

    loss = (weights * chunk_losses).sum() / weights.sum().clamp_min(1e-8)

    return loss, weights, chunk_losses
    
def sample_air3d_candidates(train_dataset, n, device, independent_tau=True):
    x_phys = train_dataset._sample_states_phys(n)
    x_net = train_dataset._scale_states(x_phys)

    if train_dataset.tau_max <= 0.0:
        tau_phys = torch.zeros(n, 1)
    elif independent_tau:
        tau_phys = train_dataset.tau_max * torch.rand(n, 1)
    else:
        tau_phys = train_dataset._sample_tau_phys(n)

    tau_net = train_dataset._scale_tau(tau_phys)
    xt = torch.cat([x_net, tau_net], dim=-1)

    return {
        "xt": xt.to(device).float().requires_grad_(True),
        "x_phys": x_phys.to(device).float(),
        "tau_phys": tau_phys.squeeze(-1).to(device).float(),
    }


def build_residual_distribution_pool(
    model,
    residual,
    train_dataset,
    device,
    num_candidates,
    chunk_size=2048,
    rad_k=1.0,
    rad_c=1.0,
    rad_eps=1e-8,
    independent_tau=True,
):
    was_training = model.training
    model.eval()

    xt_list = []
    x_phys_list = []
    tau_phys_list = []
    score_list = []

    remaining = int(num_candidates)

    while remaining > 0:
        n = min(int(chunk_size), remaining)
        remaining -= n

        cand = sample_air3d_candidates(
            train_dataset=train_dataset,
            n=n,
            device=device,
            independent_tau=independent_tau,
        )

        xt = cand["xt"]
        tau_phys = cand["tau_phys"]

        V, raw, latent, dlatent = model.compute_value_at_xt(xt, tau_phys)

        r = residual.compute_cnf_rom_residual(
            model=model,
            V=V,
            raw=raw,
            xt=xt,
            latent=latent,
            dlatent=dlatent,
        )

        scores = r.abs().detach()

        xt_list.append(cand["xt"].detach().cpu())
        x_phys_list.append(cand["x_phys"].detach().cpu())
        tau_phys_list.append(cand["tau_phys"].detach().cpu())
        score_list.append(scores.cpu())

        del cand, xt, tau_phys, V, raw, latent, dlatent, r, scores

    if was_training:
        model.train()

    xt_all = torch.cat(xt_list, dim=0)
    x_phys_all = torch.cat(x_phys_list, dim=0)
    tau_phys_all = torch.cat(tau_phys_list, dim=0)
    scores = torch.cat(score_list, dim=0)

    score_power = scores.pow(float(rad_k))
    weights = score_power / score_power.mean().clamp_min(rad_eps)
    weights = weights + float(rad_c)

    weight_sum = weights.sum()

    if (not torch.isfinite(weight_sum)) or weight_sum.item() <= 0:
        probs = torch.full_like(weights, 1.0 / weights.numel())
    else:
        probs = weights / weight_sum

    return {
        "xt": xt_all,
        "x_phys": x_phys_all,
        "tau_phys": tau_phys_all,
        "scores": scores,
        "probs": probs,
    }


def sample_from_residual_distribution(pool, num_samples, replace=False):
    num_samples = int(num_samples)

    if num_samples > pool["probs"].numel():
        replace = True

    idx = torch.multinomial(
        pool["probs"],
        num_samples=num_samples,
        replacement=replace,
    )

    return {
        "xt": pool["xt"][idx].float(),
        "x_phys": pool["x_phys"][idx].float(),
        "tau_phys": pool["tau_phys"][idx].float(),
    }


def replace_interior_points(batch, sampled):
    batch["xt_interior"] = sampled["xt"]
    batch["x_interior_phys"] = sampled["x_phys"]
    batch["tau_interior_phys"] = sampled["tau_phys"]
    return batch


class RARDAnchorSet:
    def __init__(self, max_anchors=None):
        self.max_anchors = max_anchors
        self.xt = None
        self.x_phys = None
        self.tau_phys = None

    def add(self, sampled):
        if self.xt is None:
            self.xt = sampled["xt"]
            self.x_phys = sampled["x_phys"]
            self.tau_phys = sampled["tau_phys"]
        else:
            self.xt = torch.cat([self.xt, sampled["xt"]], dim=0)
            self.x_phys = torch.cat([self.x_phys, sampled["x_phys"]], dim=0)
            self.tau_phys = torch.cat([self.tau_phys, sampled["tau_phys"]], dim=0)

        if self.max_anchors is not None and self.xt.shape[0] > self.max_anchors:
            keep = torch.arange(self.xt.shape[0] - self.max_anchors, self.xt.shape[0])
            self.xt = self.xt[keep]
            self.x_phys = self.x_phys[keep]
            self.tau_phys = self.tau_phys[keep]

    def num_anchors(self):
        if self.xt is None:
            return 0
        return self.xt.shape[0]

    def append_to_batch(self, batch):
        if self.xt is None:
            return batch

        batch["xt_interior"] = torch.cat(
            [batch["xt_interior"], self.xt],
            dim=0,
        )

        batch["x_interior_phys"] = torch.cat(
            [batch["x_interior_phys"], self.x_phys],
            dim=0,
        )

        batch["tau_interior_phys"] = torch.cat(
            [batch["tau_interior_phys"], self.tau_phys],
            dim=0,
        )

        return batch