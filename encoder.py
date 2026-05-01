"""Encoder for the optical WM JEPA.

Option A (PoC-safe): Conv1D spectral branch + masked LP MLP + fusion.
No GNN — topology is fixed (11 links) and pooling captures enough.

Input (one timestep):
  spectral_occupancy : [B, 11, 40]   bool  → cast to float
  channel_gsnr_db    : [B, 11, 40]   float32
  channel_nli_dbm    : [B, 11, 40]   float32
  lp_table           : [B, 40, 4]    float32  (slot, src, dst, n_hops)
  lp_active          : [B, 40]       bool

Output: z [B, D_z]  (D_z=128 by default)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEncoder(nn.Module):
    """Per-link Conv1D along the slot dimension, then pool across links.

    Each link gets its own 3-channel signal [occupancy, gsnr, nli] over 40 slots.
    Conv1D captures local slot-coupling patterns (XPM between adjacent slots).
    """

    def __init__(self, n_links: int = 11, n_slots: int = 40,
                 hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.n_links = n_links
        self.n_slots = n_slots

        # 3 input channels: occupancy (float), gsnr_normalized, nli_normalized
        self.conv = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # After pool: [B*n_links, hidden] → aggregate across links
        self.link_proj = nn.Linear(hidden, out_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, occupancy: torch.Tensor, gsnr: torch.Tensor,
                nli: torch.Tensor) -> torch.Tensor:
        """
        Args:
            occupancy: [B, n_links, n_slots] float (0/1)
            gsnr:      [B, n_links, n_slots] float (normalized, 0 on free slots)
            nli:       [B, n_links, n_slots] float (normalized, 0 on free slots)
        Returns:
            [B, out_dim]
        """
        B = occupancy.shape[0]

        # Stack channels: [B, 3, n_links, n_slots]
        x = torch.stack([occupancy, gsnr, nli], dim=1)

        # Treat each link independently: reshape to [B*n_links, 3, n_slots]
        x = x.permute(0, 2, 1, 3).reshape(B * self.n_links, 3, self.n_slots)

        # Conv1D along slot dim: → [B*n_links, hidden, n_slots]
        x = self.conv(x)

        # Pool over slots: → [B*n_links, hidden]
        x = x.mean(dim=-1)

        # Per-link projection: → [B*n_links, out_dim]
        x = self.link_proj(x)

        # Reshape back and pool over links: [B, n_links, out_dim] → [B, out_dim]
        x = x.reshape(B, self.n_links, -1)
        x = x.mean(dim=1)  # mean pool: all links equally weighted
        x = self.pool_proj(x)

        return x  # [B, out_dim]


class LPEncoder(nn.Module):
    """Encode active lightpaths via MLP + masked mean pool.

    Each LP is a 4-dim vector [slot, src_idx, dst_idx, n_hops].
    We embed each independently, then pool only over active LPs.
    """

    def __init__(self, lp_dim: int = 4, hidden: int = 32, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(lp_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )
        # Normalize lp inputs for stable training
        # slot: [0, 39] → /39; src/dst: [0, 7] → /7; n_hops: [1, 4] → /4
        self.register_buffer('lp_scale',
                             torch.tensor([39., 7., 7., 4.], dtype=torch.float32))

    def forward(self, lp_table: torch.Tensor,
                lp_active: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lp_table:  [B, max_lp, 4]
            lp_active: [B, max_lp] bool
        Returns:
            [B, out_dim]
        """
        # Normalize LP features to [0, 1]
        x = lp_table / (self.lp_scale + 1e-6)  # [B, max_lp, 4]
        x = self.mlp(x)                          # [B, max_lp, out_dim]

        # Masked mean pool (ignore padding rows)
        mask = lp_active.float().unsqueeze(-1)   # [B, max_lp, 1]
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # [B, out_dim]

        return x


class Encoder(nn.Module):
    """Full encoder: spectral branch + LP branch → fused latent z.

    D_z = 128 by default. This is shared between the online encoder
    (trained) and the target encoder (EMA copy, stop-gradient).
    """

    def __init__(self, n_links: int = 11, n_slots: int = 40,
                 max_lp: int = 40, d_z: int = 128):
        super().__init__()
        spectral_out = 64
        lp_out = 32

        self.spectral = SpectralEncoder(n_links, n_slots,
                                         hidden=32, out_dim=spectral_out)
        self.lp_enc = LPEncoder(lp_dim=4, hidden=32, out_dim=lp_out)

        self.fusion = nn.Sequential(
            nn.Linear(spectral_out + lp_out, d_z),
            nn.LayerNorm(d_z),
            nn.ReLU(),
        )
        self.d_z = d_z

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys matching dataset state tensors
        Returns:
            z: [B, d_z]
        """
        occ  = batch['spectral_occupancy'].float()   # [B, 11, 40]
        gsnr = batch['channel_gsnr_db']              # [B, 11, 40]
        nli  = batch['channel_nli_dbm']              # [B, 11, 40]
        lpt  = batch['lp_table']                     # [B, 40, 4]
        mask = batch['lp_active']                    # [B, 40] bool

        z_spec = self.spectral(occ, gsnr, nli)       # [B, 64]
        z_lp   = self.lp_enc(lpt, mask)              # [B, 32]

        z = self.fusion(torch.cat([z_spec, z_lp], dim=-1))  # [B, 128]
        return z


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape check
    B = 8
    enc = Encoder(n_links=11, n_slots=40, max_lp=40, d_z=128)
    batch = {
        'spectral_occupancy': torch.randint(0, 2, (B, 11, 40)).bool(),
        'channel_gsnr_db':    torch.randn(B, 11, 40),
        'channel_nli_dbm':    torch.randn(B, 11, 40),
        'lp_table':           torch.randn(B, 40, 4),
        'lp_active':          torch.randint(0, 2, (B, 40)).bool(),
    }
    z = enc(batch)
    print(f"Encoder output shape: {z.shape}")       # [8, 128]
    print(f"Encoder params: {count_params(enc):,}") # ~85K
