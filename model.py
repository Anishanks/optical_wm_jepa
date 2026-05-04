"""
JEPA World Model for optical networks (stable version with residual prediction)

Key fix: predictor outputs the DELTA (z_target - z_t) instead of the absolute
z_target. This addresses the dataset's low signal-to-noise ratio at short
horizons, where most transitions barely change the latent — without the
residual formulation, the predictor injects noise rather than learning the
small but meaningful delta.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .predictor import Predictor


class JEPAWorldModel(nn.Module):

    def __init__(self, n_links=11, n_slots=40,
                 max_lp=40, d_z=128,
                 ema_tau=0.99):

        super().__init__()

        self.d_z = d_z
        self.ema_tau = ema_tau

        # Online encoder
        self.encoder = Encoder(n_links, n_slots, max_lp, d_z)

        # Target encoder (EMA)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(d_z=d_z)

    # =====================================================
    # EMA update (called externally each step)
    # =====================================================
    @torch.no_grad()
    def update_target_encoder(self):
        for p_o, p_t in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_t.data.mul_(self.ema_tau).add_(
                p_o.data, alpha=1.0 - self.ema_tau
            )

    # =====================================================
    # TRAIN STEP
    # =====================================================
    def forward(self, s_t, actions, s_tk):

        # online encoding
        z_t = self.encoder(s_t)

        # target encoding (stop-gradient via no_grad)
        with torch.no_grad():
            z_target = self.target_encoder(s_tk)

        # 🔥 RESIDUAL PREDICTION
        # Predictor outputs delta_z, we add z_t to get z_pred.
        # Equivalent to learning MSE(delta_pred, z_target - z_t).
        # If the predictor outputs zero, z_pred = z_t = baseline,
        # which is a sensible default for low-change transitions.
        delta_z = self.predictor(z_t, actions)
        z_pred = z_t + delta_z

        # raw MSE in latent space (no F.normalize — interferes with VICReg)
        loss = F.mse_loss(z_pred, z_target)

        return {
            "loss": loss,
            "z_t": z_t,
            "z_pred": z_pred,
            "z_target": z_target,
            "delta_z": delta_z,   # exposed for diagnostics
        }

    # =====================================================
    # inference
    # =====================================================
    def encode(self, s):
        with torch.no_grad():
            return self.encoder(s)

    def predict(self, z_t, actions):
        """Inference: returns the absolute predicted latent."""
        with torch.no_grad():
            delta_z = self.predictor(z_t, actions)
            return z_t + delta_z


# =====================================================
# VICReg (kept identical)
# =====================================================
def vicreg_loss(z, lambda_var=25.0, lambda_cov=1.0):

    B, D = z.shape
    z = z - z.mean(dim=0)

    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1 - std).mean()

    cov = (z.T @ z) / (B - 1)
    cov = cov.fill_diagonal_(0)
    cov_loss = (cov ** 2).sum() / D

    return lambda_var * var_loss + lambda_cov * cov_loss


# =====================================================
# Utils
# =====================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)