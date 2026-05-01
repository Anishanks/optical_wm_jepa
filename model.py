"""
JEPA World Model for optical networks (stable version)
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
    # EMA update (MUST be called externally each step)
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

        # target encoding (fixed graph)
        with torch.no_grad():
            z_target = self.target_encoder(s_tk)

        # prediction
        z_pred = self.predictor(z_t, actions)

        # normalized MSE (IMPORTANT for stability)
        z_pred = F.normalize(z_pred, dim=-1)
        z_target = F.normalize(z_target, dim=-1)

        loss = F.mse_loss(z_pred, z_target)

        return {
            "loss": loss,
            "z_t": z_t,
            "z_pred": z_pred,
            "z_target": z_target
        }

    # =====================================================
    # inference
    # =====================================================
    def encode(self, s):
        with torch.no_grad():
            return self.encoder(s)

    def predict(self, z_t, actions):
        with torch.no_grad():
            return self.predictor(z_t, actions)


# =====================================================
# VICReg (OK version)
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