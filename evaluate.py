"""Evaluation for the JEPA world model.

Three axes:
  1. Probing   : linear probes from frozen z_t → physical quantities
  2. Rollout   : multi-step prediction error vs horizon h
  3. Collapse  : latent space health diagnostics

Usage:
    python -m optical_wm_jepa.evaluate \
        --checkpoint checkpoints/best_model.pt \
        --dataset data/optical_wm_poc.h5 \
        --norm-stats data/norm_stats.json
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from .model import JEPAWorldModel
from .train import HDF5BatchSampler, normalize_batch, load_norm_stats


# =====================================================================
# Utilities
# =====================================================================

def load_model(checkpoint_path: str, device: torch.device) -> JEPAWorldModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['config']
    model = JEPAWorldModel(
        n_links=cfg.get('n_links', 11),
        n_slots=cfg.get('n_slots', 40),
        max_lp=cfg.get('max_lp', 40),
        d_z=cfg.get('d_z', 128),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def collect_latents_and_targets(
    model: JEPAWorldModel,
    sampler: HDF5BatchSampler,
    norm_stats: dict,
    device: torch.device,
    n_samples: int = 5000,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Collect (z_t, physical targets) over n_samples timesteps."""
    rng = np.random.default_rng(0)
    zs = []
    n_actives = []
    mean_gsnrs = []
    per_lp_gsnrs = []    # [N, max_lp]
    per_lp_nlis = []     # [N, max_lp]   (derived from gsnr and nli arrays)
    lp_masks = []

    collected = 0
    while collected < n_samples:
        bs = min(batch_size, n_samples - collected)
        raw = sampler.sample_batch(bs, k=1, rng=rng)
        s_t = normalize_batch(raw['s_t'], norm_stats, device)

        with torch.no_grad():
            z = model.encode(s_t)  # [B, d_z]

        zs.append(z.cpu().numpy())

        # Physical targets — computed from raw (un-normalized) values
        occ  = raw['s_t']['spectral_occupancy'].astype(float)  # [B, 11, 40]
        gsnr = raw['s_t']['channel_gsnr_db']                   # [B, 11, 40]
        nli  = raw['s_t']['channel_nli_dbm']                   # [B, 11, 40]
        lpt  = raw['s_t']['lp_table']                          # [B, 40, 4]
        mask = raw['s_t']['lp_active']                         # [B, 40] bool

        n_actives.append(mask.sum(axis=1))  # [B]
        mean_gsnrs.append(
            np.where(occ.sum(axis=(1, 2)) > 0,
                     (gsnr * occ).sum(axis=(1, 2)) / (occ.sum(axis=(1, 2)) + 1e-9),
                     0.0)
        )  # [B]

        # Per-LP GSNR: look up the LP's slot on any link it traverses
        B = bs
        per_lp_g = np.zeros((B, lpt.shape[1]))
        per_lp_n = np.zeros((B, lpt.shape[1]))
        for b in range(B):
            for i in range(lpt.shape[1]):
                if not mask[b, i]:
                    continue
                slot = int(lpt[b, i, 0])
                if 0 <= slot < gsnr.shape[2]:
                    # Take mean GSNR over occupied link positions
                    vals = gsnr[b, occ[b, :, slot].astype(bool), slot]
                    if len(vals) > 0:
                        per_lp_g[b, i] = vals.mean()
                    vals_n = nli[b, occ[b, :, slot].astype(bool), slot]
                    if len(vals_n) > 0:
                        per_lp_n[b, i] = vals_n.mean()
        per_lp_gsnrs.append(per_lp_g)
        per_lp_nlis.append(per_lp_n)
        lp_masks.append(mask)

        collected += bs

    return {
        'z':            np.concatenate(zs, axis=0),
        'n_active':     np.concatenate(n_actives, axis=0),
        'mean_gsnr':    np.concatenate(mean_gsnrs, axis=0),
        'per_lp_gsnr':  np.concatenate(per_lp_gsnrs, axis=0),
        'per_lp_nli':   np.concatenate(per_lp_nlis, axis=0),
        'lp_mask':      np.concatenate(lp_masks, axis=0),
    }


# =====================================================================
# Axis 1: Probing
# =====================================================================

def run_probing(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Train linear Ridge regressors from z → physical targets.

    The R² tells us: how much of the variance in each physical quantity
    is linearly accessible from the latent code z?

    A good WM should score:
      n_active:   R² > 0.95  (sanity — easy)
      mean_gsnr:  R² > 0.80  (aggregate quality)
      per_lp_gsnr: R² > 0.70 (per-connection quality)
      per_lp_nli:  R² > 0.50 (NLI coupling — the key discriminating probe)
    """
    z = data['z']  # [N, d_z]
    N = z.shape[0]
    split = int(0.8 * N)
    z_tr, z_te = z[:split], z[split:]

    scaler = StandardScaler()
    z_tr_s = scaler.fit_transform(z_tr)
    z_te_s = scaler.transform(z_te)

    results = {}

    def probe(name: str, y: np.ndarray, mask: np.ndarray = None):
        """Train and eval one probe. mask selects valid rows."""
        if mask is not None:
            # Flatten and filter
            y_flat = y.reshape(N, -1)
            m_flat = mask.reshape(N, -1)
            # Use mean over active LPs per sample
            valid = m_flat.sum(axis=1) > 0
            y_agg = np.where(valid[:, None],
                             (y_flat * m_flat).sum(axis=1, keepdims=True)
                             / (m_flat.sum(axis=1, keepdims=True) + 1e-9),
                             0.0).squeeze(1)
        else:
            y_agg = y

        y_tr = y_agg[:split]
        y_te = y_agg[split:]

        reg = Ridge(alpha=1.0)
        reg.fit(z_tr_s, y_tr)
        y_pred = reg.predict(z_te_s)
        r2 = r2_score(y_te, y_pred)
        results[name] = float(r2)
        return r2

    r2_n  = probe('n_active',  data['n_active'])
    r2_g  = probe('mean_gsnr', data['mean_gsnr'])
    r2_lg = probe('per_lp_gsnr_mean', data['per_lp_gsnr'], mask=data['lp_mask'])
    r2_nl = probe('per_lp_nli_mean',  data['per_lp_nli'],  mask=data['lp_mask'])

    print("\n=== Probing Results ===")
    targets = [
        ('n_active',          r2_n,  0.95, 'sanity'),
        ('mean_gsnr',         r2_g,  0.80, 'aggregate quality'),
        ('per_lp_gsnr (avg)', r2_lg, 0.70, 'per-LP quality'),
        ('per_lp_nli (avg)',  r2_nl, 0.50, 'NLI coupling — key probe'),
    ]
    for name, r2, target, note in targets:
        ok = "PASS" if r2 >= target else "FAIL"
        print(f"  [{ok}] {name:22s} R²={r2:.3f}  (target ≥ {target})  ← {note}")

    return results


# =====================================================================
# Axis 2: Multi-step rollout
# =====================================================================

def run_rollout_eval(
    model: JEPAWorldModel,
    sampler: HDF5BatchSampler,
    norm_stats: dict,
    device: torch.device,
    horizons: List[int] = (1, 2, 5, 10, 25),
    n_samples: int = 500,
) -> Dict[int, Dict[str, float]]:
    """Measure prediction error vs horizon h.

    For each h:
      - Sample (s_t, actions[t:t+h], s_{t+h}) from val set
      - Roll predictor h steps in latent
      - Measure: relative latent error, and MAE of a linear GSNR decoder

    Baseline: z_pred = z_t (no change predicted) — trivially wrong for h>1.
    """
    rng = np.random.default_rng(1)
    results = {}

    # Quick linear decoder: z → mean_gsnr (fit on separate data)
    print("\n=== Rollout Evaluation ===")
    print(f"  Fitting linear GSNR decoder...")
    calib_data = collect_latents_and_targets(
        model, sampler, norm_stats, device, n_samples=2000
    )
    z_cal = calib_data['z']
    y_cal = calib_data['mean_gsnr']
    scaler = StandardScaler()
    z_cal_s = scaler.fit_transform(z_cal)
    decoder = Ridge(alpha=1.0).fit(z_cal_s, y_cal)

    for h in horizons:
        lat_errs = []
        baseline_errs = []
        gsnr_maes = []
        gsnr_baseline_maes = []

        for _ in range(max(1, n_samples // 64)):
            raw = sampler.sample_batch(64, k=h, rng=rng)
            s_t  = normalize_batch(raw['s_t'],  norm_stats, device)
            s_tk = normalize_batch(raw['s_tk'], norm_stats, device)
            actions = torch.from_numpy(raw['actions']).float().to(device)

            with torch.no_grad():
                z_t      = model.encode(s_t)
                z_target = model.encode(s_tk)
                z_pred   = model.predict(z_t, actions)

            # Relative latent error
            target_norm = (z_target ** 2).mean(dim=-1)
            pred_err    = ((z_pred - z_target) ** 2).mean(dim=-1)
            baseline_err = ((z_t   - z_target) ** 2).mean(dim=-1)

            lat_errs.append((pred_err / (target_norm + 1e-6)).mean().item())
            baseline_errs.append((baseline_err / (target_norm + 1e-6)).mean().item())

            # GSNR MAE via linear decoder
            z_pred_np   = z_pred.cpu().numpy()
            z_target_np = z_target.cpu().numpy()
            z_t_np      = z_t.cpu().numpy()

            gsnr_true = decoder.predict(scaler.transform(z_target_np))
            gsnr_pred = decoder.predict(scaler.transform(z_pred_np))
            gsnr_base = decoder.predict(scaler.transform(z_t_np))

            gsnr_maes.append(np.abs(gsnr_pred - gsnr_true).mean())
            gsnr_baseline_maes.append(np.abs(gsnr_base - gsnr_true).mean())

        results[h] = {
            'lat_err_relative': float(np.mean(lat_errs)),
            'baseline_lat_err': float(np.mean(baseline_errs)),
            'gsnr_mae_db':      float(np.mean(gsnr_maes)),
            'baseline_mae_db':  float(np.mean(gsnr_baseline_maes)),
        }
        r = results[h]
        ratio = r['lat_err_relative'] / (r['baseline_lat_err'] + 1e-6)
        print(f"  h={h:2d}: latent_err={r['lat_err_relative']:.4f} "
              f"(baseline={r['baseline_lat_err']:.4f}, ratio={ratio:.2f}) | "
              f"GSNR_MAE={r['gsnr_mae_db']:.3f} dB "
              f"(baseline={r['baseline_mae_db']:.3f} dB)")

    print("\n  A good WM: ratio < 0.5 for h≤5, < 0.8 for h≤25")
    print("  (ratio = pred_error / baseline_error — lower is better)")
    return results


# =====================================================================
# Axis 3: Latent space diagnostics
# =====================================================================

def run_latent_diagnostics(data: Dict[str, np.ndarray]):
    """Check latent space health after training."""
    z = data['z']
    std_per_dim = z.std(axis=0)
    mean_per_dim = z.mean(axis=0)

    print("\n=== Latent Space Diagnostics ===")
    print(f"  d_z:         {z.shape[1]}")
    print(f"  N samples:   {z.shape[0]}")
    print(f"  std per dim: mean={std_per_dim.mean():.3f}, "
          f"min={std_per_dim.min():.3f}, max={std_per_dim.max():.3f}")
    print(f"  dead dims (std<0.05):  {(std_per_dim < 0.05).sum()}")
    print(f"  near-dead (std<0.10):  {(std_per_dim < 0.10).sum()}")

    # Effective rank (how many dims carry variance)
    var = std_per_dim ** 2
    var_norm = var / (var.sum() + 1e-9)
    entropy = -(var_norm * np.log(var_norm + 1e-9)).sum()
    eff_rank = int(np.exp(entropy))
    print(f"  effective rank:        {eff_rank}/{z.shape[1]}")

    if std_per_dim.min() < 0.05:
        print("  ⚠ Warning: collapsed dims detected → consider enabling VICReg")
    elif eff_rank < z.shape[1] // 3:
        print("  ⚠ Warning: low effective rank → latent may be underdetermined")
    else:
        print("  ✓ Latent space looks healthy")


# =====================================================================
# Full eval
# =====================================================================

def evaluate(
    checkpoint_path: str,
    dataset_path: str,
    norm_stats_path: str,
    output_dir: str = 'checkpoints',
    device_str: str = 'auto',
):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    print(f"=== JEPA World Model Evaluation ===")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")

    model = load_model(checkpoint_path, device)
    norm_stats = load_norm_stats(norm_stats_path)
    val_sampler = HDF5BatchSampler(dataset_path, split='val')

    # Collect latents from val set
    print("\nCollecting val latents...")
    data = collect_latents_and_targets(
        model, val_sampler, norm_stats, device, n_samples=3000
    )

    # Run all axes
    probe_results  = run_probing(data)
    rollout_results = run_rollout_eval(model, val_sampler, norm_stats, device)
    run_latent_diagnostics(data)

    # Save results
    out = {
        'checkpoint': checkpoint_path,
        'probing': probe_results,
        'rollout': {str(h): v for h, v in rollout_results.items()},
    }
    out_path = Path(output_dir) / 'eval_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    val_sampler.close()
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  default='checkpoints/best_model.pt')
    p.add_argument('--dataset',     default='data/optical_wm_poc.h5')
    p.add_argument('--norm-stats',  default='data/norm_stats.json')
    p.add_argument('--output',      default='checkpoints')
    p.add_argument('--device',      default='auto')
    args = p.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        norm_stats_path=args.norm_stats,
        output_dir=args.output,
        device_str=args.device,
    )
