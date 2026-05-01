"""Training loop for the JEPA world model.

Key design decisions:
  - k-curriculum: start with k=1 only, progressively introduce larger k
  - Collapse monitor: track std(z) per dimension each epoch
  - VICReg toggle: off by default, enable via flag if collapse detected
  - Normalization: applied at load time using precomputed norm_stats.json
"""
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import JEPAWorldModel, vicreg_loss, count_params


# =====================================================================
# Normalization helper
# =====================================================================

def load_norm_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def normalize_batch(batch: dict, stats: dict, device: torch.device) -> dict:
    """Normalize channel_gsnr_db and channel_nli_dbm using masked stats.
    Free slots stay at 0; only occupied positions are shifted/scaled.
    """
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        out[k] = v.to(device)

    for key in ['channel_gsnr_db', 'channel_nli_dbm']:
        if key not in stats:
            continue
        s = stats[key]
        mask = out['spectral_occupancy'].float()  # 1 on occupied
        normalized = torch.zeros_like(out[key])
        normalized[mask.bool()] = (
            (out[key][mask.bool()] - s['mean']) / (s['std'] + 1e-6)
        )
        out[key] = normalized

    return out


# =====================================================================
# k-curriculum
# =====================================================================

K_CURRICULUM = {
    # epoch range: k values and their sampling weights
    'warm_up':  {'range': (0,  10), 'k_weights': {1: 1.0}},
    'phase1':   {'range': (10, 30), 'k_weights': {1: 0.5, 2: 0.3, 5: 0.2}},
    'phase2':   {'range': (30, 60), 'k_weights': {1: 0.3, 2: 0.2, 5: 0.3, 10: 0.2}},
    'phase3':   {'range': (60, 100), 'k_weights': {1: 0.2, 2: 0.15, 5: 0.25, 10: 0.25, 25: 0.15}},
}


def sample_k(epoch: int, rng: np.random.Generator) -> int:
    """Sample k according to curriculum at current epoch."""
    for phase in K_CURRICULUM.values():
        start, end = phase['range']
        if start <= epoch < end:
            kv = phase['k_weights']
            ks = list(kv.keys())
            ws = list(kv.values())
            return int(rng.choice(ks, p=ws))
    return 1  # fallback


# =====================================================================
# Batch sampler from HDF5
# =====================================================================

class HDF5BatchSampler:
    """Simple batch sampler that reads directly from HDF5.

    For a proper DataLoader with workers, wrap this in a torch Dataset.
    This simpler version works well for PoC training on CPU or single GPU.
    """

    def __init__(self, filepath: str, split: str = 'train'):
        import h5py
        self.filepath = filepath
        self.split = split
        self._f = h5py.File(filepath, 'r')

        split_data = self._f['split']
        self.episode_ids = [
            s.decode() if isinstance(s, bytes) else s
            for s in split_data[split][:]
        ]
        self.episode_lengths = {
            ep: int(self._f[f'episodes/{ep}'].attrs['n_steps'])
            for ep in self.episode_ids
        }

    def sample_batch(self, batch_size: int, k: int,
                     rng: np.random.Generator) -> dict:
        """Sample a batch of (s_t, actions[k], s_{t+k}) from random episodes."""
        s_t_list = {key: [] for key in [
            'spectral_occupancy', 'channel_gsnr_db', 'channel_nli_dbm',
            'lp_table', 'lp_active',
        ]}
        s_tk_list = {key: [] for key in s_t_list}
        actions_list = []

        # Filter episodes long enough
        eligible = [eid for eid, n in self.episode_lengths.items() if n > k + 1]
        if not eligible:
            raise RuntimeError(f"No episode long enough for k={k}")

        for _ in range(batch_size):
            ep_id = rng.choice(eligible)
            n_steps = self.episode_lengths[ep_id]
            t = int(rng.integers(0, n_steps - k - 1))

            ep = self._f[f'episodes/{ep_id}']
            st = ep['states']

            for key in s_t_list:
                s_t_list[key].append(st[key][t])
                s_tk_list[key].append(st[key][t + k])

            actions_list.append(ep['actions'][t:t + k])

        # Stack
        s_t = {k_: np.stack(v) for k_, v in s_t_list.items()}
        s_tk = {k_: np.stack(v) for k_, v in s_tk_list.items()}
        actions = np.stack(actions_list)  # [B, k, 8]

        return {'s_t': s_t, 'actions': actions, 's_tk': s_tk}

    def close(self):
        self._f.close()


# =====================================================================
# Collapse monitor
# =====================================================================

def check_collapse(zs: torch.Tensor, threshold: float = 0.10) -> dict:
    """Check if latent dimensions have collapsed (std too low).

    Args:
        zs: collected z_t values over one epoch [N, d_z]
        threshold: std per dim below this → collapse warning

    Returns:
        dict with 'collapsed', 'mean_std', 'min_std', 'dead_dims'
    """
    std_per_dim = zs.std(dim=0)  # [d_z]
    dead = (std_per_dim < threshold).sum().item()
    return {
        'collapsed': dead > zs.shape[1] // 4,  # >25% dead dims = collapse
        'mean_std': std_per_dim.mean().item(),
        'min_std': std_per_dim.min().item(),
        'dead_dims': int(dead),
    }


# =====================================================================
# Main training loop
# =====================================================================

def train(
    dataset_path: str,
    norm_stats_path: str,
    output_dir: str = 'checkpoints',
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    ema_tau: float = 0.99,
    d_z: int = 128,
    use_vicreg: bool = False,
    vicreg_lambda: float = 1.0,
    log_every: int = 5,
    device_str: str = 'auto',
):
    # ── Setup ──
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # ── Data ──
    norm_stats = load_norm_stats(norm_stats_path)
    sampler = HDF5BatchSampler(dataset_path, split='train')
    val_sampler = HDF5BatchSampler(dataset_path, split='val')
    print(f"Train episodes: {len(sampler.episode_ids)}, "
          f"Val: {len(val_sampler.episode_ids)}")

    # ── Model ──
    model = JEPAWorldModel(d_z=d_z, ema_tau=ema_tau).to(device)
    n_params = count_params(model.encoder) + count_params(model.predictor)
    print(f"Trainable params: {n_params:,}")

    # ── Optimizer ──
    optimizer = AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.1)

    # ── Training ──
    history = []
    best_val_loss = float('inf')

    steps_per_epoch = max(1, len(sampler.episode_ids) * 60 // batch_size)

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        z_buffer = []      # for collapse monitoring

        t0 = time.time()
        for _ in range(steps_per_epoch):
            k = sample_k(epoch, rng)
            raw = sampler.sample_batch(batch_size, k, rng)

            s_t  = normalize_batch(raw['s_t'],  norm_stats, device)
            s_tk = normalize_batch(raw['s_tk'], norm_stats, device)
            actions = torch.from_numpy(raw['actions']).float().to(device)

            optimizer.zero_grad()
            out = model(s_t, actions, s_tk)
            loss = out['loss']

            # Optional VICReg
            if use_vicreg:
                loss = loss + vicreg_lambda * vicreg_loss(out['z_t'])

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_encoder()

            epoch_losses.append(loss.item())
            z_buffer.append(out['z_t'].cpu())

        scheduler.step()

        # ── Validation loss ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(max(1, len(val_sampler.episode_ids) // 4)):
                k_val = 5  # fixed k for val
                raw_val = val_sampler.sample_batch(
                    min(batch_size, 64), k_val, rng
                )
                sv_t  = normalize_batch(raw_val['s_t'],  norm_stats, device)
                sv_tk = normalize_batch(raw_val['s_tk'], norm_stats, device)
                av    = torch.from_numpy(raw_val['actions']).float().to(device)
                val_out = model(sv_t, av, sv_tk)
                val_losses.append(val_out['loss'].item())

        # ── Collapse check ──
        z_all = torch.cat(z_buffer, dim=0)
        collapse = check_collapse(z_all)

        train_loss = float(np.mean(epoch_losses))
        val_loss   = float(np.mean(val_losses))

        log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0],
            'k_max': sample_k(epoch, rng),
            'mean_std_z': collapse['mean_std'],
            'dead_dims': collapse['dead_dims'],
            'elapsed_s': time.time() - t0,
        }
        history.append(log)

        # ── Logging ──
        if (epoch + 1) % log_every == 0:
            collapse_warn = " ⚠ COLLAPSE" if collapse['collapsed'] else ""
            print(
                f"Epoch {epoch+1:3d}/{n_epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} | "
                f"lr={log['lr']:.2e} | "
                f"std(z)={collapse['mean_std']:.3f} dead={collapse['dead_dims']}"
                + collapse_warn
            )
            if collapse['collapsed']:
                print("  → Add VICReg: set use_vicreg=True in next run")

        # ── Checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {'d_z': d_z, 'n_links': 11, 'n_slots': 40, 'max_lp': 40},
            }, f"{output_dir}/best_model.pt")

    # Save history
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    sampler.close()
    val_sampler.close()
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_dir}/best_model.pt")
    return history


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    default='data/optical_wm_poc.h5')
    p.add_argument('--norm-stats', default='data/norm_stats.json')
    p.add_argument('--output',     default='checkpoints')
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch-size', type=int,   default=256)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--d-z',        type=int,   default=128)
    p.add_argument('--vicreg',     action='store_true')
    p.add_argument('--device',     default='auto')
    args = p.parse_args()

    train(
        dataset_path=args.dataset,
        norm_stats_path=args.norm_stats,
        output_dir=args.output,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_z=args.d_z,
        use_vicreg=args.vicreg,
        device_str=args.device,
    )
