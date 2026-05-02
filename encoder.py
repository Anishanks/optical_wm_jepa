"""
Encoder for the optical WM JEPA.

Option A (PoC-safe): Conv1D spectral branch + masked LP MLP + fusion.
No GNN — topology is fixed (11 links) and pooling captures enough.

FIXES APPLIED (collapse root cause):
- BatchNorm1d after each Conv1d (prevents activation collapse on sparse input)
- mean+max pool over slots instead of mean only (preserves diversity)
- attention pool over links (instead of mean collapse)
- NO ReLU at any output (Linear → LayerNorm only at fusion)
"""

import torch
import torch.nn as nn


# =========================================================
# Spectral Encoder
# =========================================================
class SpectralEncoder(nn.Module):
    """
    Per-link Conv1D along slot dimension + attention over links.
    """

    def __init__(self, n_links=11, n_slots=40,
                 hidden=32, out_dim=64):
        super().__init__()

        self.n_links = n_links
        self.n_slots = n_slots

        # 🔥 FIX: BatchNorm1d after each Conv1d
        # Prevents activation collapse on sparse input (~90% zeros)
        self.conv = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # 🔥 FIX: link_proj input doubled because we concat mean+max
        self.link_proj = nn.Linear(2 * hidden, out_dim)

        # 🔥 attention over links (instead of mean collapse)
        self.link_attn = nn.Linear(out_dim, 1)

        # 🔥 FIX: pool_proj without ReLU (allow negative dims)
        self.pool_proj = nn.Linear(out_dim, out_dim)

    def forward(self, occupancy, gsnr, nli):
        B = occupancy.shape[0]

        # [B, 3, links, slots]
        x = torch.stack([occupancy, gsnr, nli], dim=1)

        # [B*links, 3, slots]
        x = x.permute(0, 2, 1, 3).reshape(B * self.n_links, 3, self.n_slots)

        x = self.conv(x)

        # 🔥 FIX: mean+max pool over slots (was just mean)
        # Preserves more spatial info — peaks survive averaging
        x_mean = x.mean(dim=-1)
        x_max  = x.max(dim=-1).values
        x = torch.cat([x_mean, x_max], dim=-1)   # [B*links, 2*hidden]

        x = self.link_proj(x)

        # reshape → [B, links, out_dim]
        x = x.view(B, self.n_links, -1)

        # 🔥 attention pooling (important fix)
        attn = self.link_attn(x)                  # [B, links, 1]
        attn = torch.softmax(attn, dim=1)

        x = (x * attn).sum(dim=1)                 # [B, out_dim]

        x = self.pool_proj(x)

        return x


# =========================================================
# LP Encoder
# =========================================================
class LPEncoder(nn.Module):
    def __init__(self, lp_dim=4, hidden=32, out_dim=32):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(lp_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

        self.register_buffer(
            "lp_scale",
            torch.tensor([39., 7., 7., 4.], dtype=torch.float32)
        )

    def forward(self, lp_table, lp_active):
        x = lp_table / (self.lp_scale + 1e-6)

        x = self.mlp(x)

        mask = lp_active.float().unsqueeze(-1)

        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)

        return x


# =========================================================
# Full Encoder
# =========================================================
class Encoder(nn.Module):
    def __init__(self, n_links=11, n_slots=40,
                 max_lp=40, d_z=128):
        super().__init__()

        spectral_out = 64
        lp_out = 32

        self.spectral = SpectralEncoder(
            n_links=n_links,
            n_slots=n_slots,
            hidden=32,
            out_dim=spectral_out
        )

        self.lp_enc = LPEncoder(lp_dim=4, hidden=32, out_dim=lp_out)

        self.fusion = nn.Sequential(
            nn.Linear(spectral_out + lp_out, d_z),
            nn.LayerNorm(d_z),
            # ❌ IMPORTANT: NO ReLU (prevents dead dimensions)
        )

        self.d_z = d_z

    def forward(self, batch):
        occ = batch['spectral_occupancy'].float()
        gsnr = batch['channel_gsnr_db']
        nli = batch['channel_nli_dbm']
        lpt = batch['lp_table']
        mask = batch['lp_active']

        z_spec = self.spectral(occ, gsnr, nli)
        z_lp = self.lp_enc(lpt, mask)

        z = torch.cat([z_spec, z_lp], dim=-1)

        z = self.fusion(z)

        return z


# =========================================================
# Utils
# =========================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    B = 8

    enc = Encoder()

    batch = {
        'spectral_occupancy': torch.randint(0, 2, (B, 11, 40)).float(),
        'channel_gsnr_db': torch.randn(B, 11, 40),
        'channel_nli_dbm': torch.randn(B, 11, 40),
        'lp_table': torch.randn(B, 40, 4),
        'lp_active': torch.randint(0, 2, (B, 40)).bool(),
    }

    z = enc(batch)

    print("Output:", z.shape)
    print("Params:", count_params(enc))