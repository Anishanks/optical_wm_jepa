"""JEPA World Model for optical networks.

Combines:
  - Online encoder  (trained via backprop)
  - Target encoder  (EMA copy of online, stop-gradient)
  - Predictor       (maps z_t + actions → ẑ_{t+k})

Loss (Phase 1 — PoC safe):
  MSE(ẑ_{t+k}, sg(z̃_{t+k}))
  where sg = stop gradient (target encoder output)

VICReg is available as an optional add-on if collapse is observed.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .predictor import Predictor


class JEPAWorldModel(nn.Module):

    def __init__(self, n_links: int = 11, n_slots: int = 40,
                 max_lp: int = 40, d_z: int = 128,
                 ema_tau: float = 0.99):
        super().__init__()
        self.d_z = d_z
        self.ema_tau = ema_tau

        # Online encoder (receives gradients)
        self.encoder = Encoder(n_links, n_slots, max_lp, d_z)

        # Target encoder (EMA, no gradients)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(d_z=d_z)

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update: target ← τ·target + (1-τ)·online."""
        for p_online, p_target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(self.ema_tau).add_(
                p_online.data, alpha=1.0 - self.ema_tau
            )

    def forward(self, s_t: dict, actions: torch.Tensor,
                s_tk: dict) -> dict:
        """One training step forward pass.

        Args:
            s_t:     state at time t         (dict of tensors)
            actions: actions t → t+k-1       [B, k, 8]
            s_tk:    state at time t+k        (dict of tensors)

        Returns:
            dict with 'loss', 'z_t', 'z_pred', 'z_target'
        """
        # Online encoder: compute z_t
        z_t = self.encoder(s_t)             # [B, d_z]

        # Target encoder: compute z̃_{t+k} (stop-gradient via no_grad)
        with torch.no_grad():
            z_target = self.target_encoder(s_tk)  # [B, d_z]

        # Predictor: ẑ_{t+k} from z_t and action sequence
        z_pred = self.predictor(z_t, actions)     # [B, d_z]

        # MSE loss in latent space (normalized for stability)
        loss = F.mse_loss(z_pred, z_target)

        return {
            'loss': loss,
            'z_t': z_t.detach(),
            'z_pred': z_pred.detach(),
            'z_target': z_target.detach(),
        }

    def encode(self, s: dict) -> torch.Tensor:
        """Encode a state dict → latent z. Used at eval/planning time."""
        with torch.no_grad():
            return self.encoder(s)

    def predict(self, z_t: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """Predict latent at t+k. Used for rollout and planning."""
        with torch.no_grad():
            return self.predictor(z_t, actions)


# =====================================================================
# Optional: VICReg regularization (add if collapse is observed)
# =====================================================================

def vicreg_loss(z: torch.Tensor,
                lambda_var: float = 25.0,
                lambda_cov: float = 1.0) -> torch.Tensor:
    """VICReg variance + covariance terms.

    Add to MSE loss ONLY if you observe latent collapse (all z_t converge
    to a single point, loss plateaus near 0 while probes fail).

    Usage:
        loss = mse_loss + vicreg_loss(z_t)
    """
    B, D = z.shape
    z = z - z.mean(dim=0)  # center

    # Variance: each dimension should have std ≈ 1
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()

    # Covariance: dimensions should be uncorrelated
    cov = (z.T @ z) / (B - 1)
    cov = cov.fill_diagonal_(0)
    cov_loss = (cov ** 2).sum() / D

    return lambda_var * var_loss + lambda_cov * cov_loss


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from .encoder import count_params as cp_enc
    from .predictor import count_params as cp_pred

    model = JEPAWorldModel()
    B = 8

    s_t = {
        'spectral_occupancy': torch.randint(0, 2, (B, 11, 40)).bool(),
        'channel_gsnr_db':    torch.randn(B, 11, 40),
        'channel_nli_dbm':    torch.randn(B, 11, 40),
        'lp_table':           torch.randn(B, 40, 4),
        'lp_active':          torch.randint(0, 2, (B, 40)).bool(),
    }
    s_tk = {k: v.clone() for k, v in s_t.items()}  # dummy same state
    actions = torch.randn(B, 5, 8)  # k=5

    out = model(s_t, actions, s_tk)
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"z_t shape:     {out['z_t'].shape}")
    print(f"z_pred shape:  {out['z_pred'].shape}")
    print(f"z_target shape:{out['z_target'].shape}")

    enc_params  = count_params(model.encoder)
    pred_params = count_params(model.predictor)
    print(f"\nEncoder params:   {enc_params:,}")
    print(f"Predictor params: {pred_params:,}")
    print(f"Total trainable:  {enc_params + pred_params:,}")
