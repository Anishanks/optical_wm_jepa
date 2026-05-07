"""
Planning evaluation for the JEPA world model — purely in latent space.

Two complementary protocols:

  Protocol 1 — Recall@K from val trajectories
    For (s_t, s_{t+h}) pairs from the val set, the ground-truth action
    sequence a_{t..t+h-1} IS a known good plan. We check whether CEM
    rediscovers it (Recall@K) or finds something equally good.

  Protocol 3 — CEM vs Random vs Greedy
    For 100 (s_t, s_g) pairs, compare 3 planners by the latent distance
    they achieve to the goal. CEM should beat random and greedy.

Both protocols are 100% latent — no GNPy required. GNPy validation
of CEM plans is a separate evaluation script (evaluate_planning_gnpy.py).

Usage:
    python -m optical_wm_jepa.evaluate_planning \
        --checkpoint checkpoints_v3/best_model.pt \
        --dataset    data/optical_wm_poc.h5 \
        --norm-stats data/norm_stats.json \
        --output     checkpoints_v3
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch

from .model import JEPAWorldModel
from .train import HDF5BatchSampler, normalize_batch, load_norm_stats


# =====================================================================
# Helpers
# =====================================================================

def load_model(checkpoint_path: str, device: torch.device) -> JEPAWorldModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def encode_pair(model: JEPAWorldModel, s_t: dict, s_g: dict,
                norm_stats: dict, device: torch.device
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode (s_t, s_g) → (z_t, z_g). Both shape [B, d_z]."""
    s_t_n = normalize_batch(s_t, norm_stats, device)
    s_g_n = normalize_batch(s_g, norm_stats, device)
    with torch.no_grad():
        z_t = model.encode(s_t_n)
        z_g = model.encode(s_g_n)
    return z_t, z_g


def score_sequences(model: JEPAWorldModel, z_t: torch.Tensor,
                    sequences: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
    """Score N candidate action sequences for ONE (z_t, z_g) pair.

    Args:
        z_t:        [d_z]
        sequences:  [N, h, 8]   N candidate action sequences of length h
        z_g:        [d_z]
    Returns:
        scores: [N]   lower = better (squared L2 distance to goal)
    """
    N = sequences.shape[0]
    z_t_batch = z_t.unsqueeze(0).expand(N, -1)  # [N, d_z]
    with torch.no_grad():
        z_pred = model.predict(z_t_batch, sequences)  # [N, d_z]
    scores = ((z_pred - z_g.unsqueeze(0)) ** 2).sum(dim=-1)  # [N]
    return scores


# =====================================================================
# CEM planner
# =====================================================================

def cem_plan(
    model: JEPAWorldModel,
    z_t: torch.Tensor,
    z_g: torch.Tensor,
    h: int,
    n_samples: int = 200,
    n_elite: int = 20,
    n_iter: int = 5,
    device: torch.device = None,
    init_mu: torch.Tensor = None,
    init_sigma: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """Cross-Entropy Method planner in latent space.

    Iteratively refines a Gaussian over action sequences toward the
    region that minimizes ||z_pred - z_g||².

    Returns:
        best_seq: [h, 8]  the elite-mean action sequence
        best_score: float  latent distance achieved
    """
    if device is None:
        device = z_t.device

    # Initialize Gaussian distribution over action sequences
    if init_mu is None:
        mu = torch.zeros(h, 8, device=device)
    else:
        mu = init_mu.clone()
    sigma = torch.full((h, 8), init_sigma, device=device)

    best_seq = None
    best_score = float('inf')

    for it in range(n_iter):
        # Sample N sequences from current Gaussian
        noise = torch.randn(n_samples, h, 8, device=device)
        sequences = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise  # [N, h, 8]

        # Action layout heuristic: type ∈ [0, 1], others ∈ [0, 1] after norm.
        # Predictor handles flexibly thanks to its scale buffer.
        sequences = sequences.clamp(-2.0, 2.0)

        # Score
        scores = score_sequences(model, z_t, sequences, z_g)  # [N]

        # Track best
        min_score, min_idx = scores.min(dim=0)
        if min_score.item() < best_score:
            best_score = min_score.item()
            best_seq = sequences[min_idx].clone()

        # Elite update
        elite_idx = scores.topk(n_elite, largest=False).indices
        elite = sequences[elite_idx]  # [n_elite, h, 8]
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0) + 1e-3  # floor to avoid collapse

    return best_seq, best_score


# =====================================================================
# Baselines
# =====================================================================

def random_baseline(
    model: JEPAWorldModel,
    z_t: torch.Tensor,
    z_g: torch.Tensor,
    h: int,
    n_samples: int = 200,
) -> Tuple[torch.Tensor, float]:
    """Best of N random action sequences (single shot, no iteration)."""
    device = z_t.device
    sequences = torch.randn(n_samples, h, 8, device=device).clamp(-2.0, 2.0)
    scores = score_sequences(model, z_t, sequences, z_g)
    min_score, min_idx = scores.min(dim=0)
    return sequences[min_idx], min_score.item()


def greedy_baseline(
    model: JEPAWorldModel,
    z_t: torch.Tensor,
    z_g: torch.Tensor,
    h: int,
    n_candidates: int = 100,
) -> Tuple[torch.Tensor, float]:
    """Greedy 1-step planner: at each timestep, pick the action that
    minimizes distance to goal AT THAT STEP (myopic).

    Builds the sequence step by step, no lookahead.
    """
    device = z_t.device
    z_curr = z_t.clone()
    chosen = []

    for step in range(h):
        # Sample candidate single actions
        candidates = torch.randn(n_candidates, 1, 8, device=device).clamp(-2.0, 2.0)
        scores = score_sequences(model, z_curr, candidates, z_g)
        best_idx = scores.argmin().item()
        a = candidates[best_idx, 0]  # [8]
        chosen.append(a)

        # Step forward in latent
        z_curr = model.predict(
            z_curr.unsqueeze(0), a.unsqueeze(0).unsqueeze(0)
        ).squeeze(0)

    seq = torch.stack(chosen, dim=0)  # [h, 8]
    final_dist = ((z_curr - z_g) ** 2).sum().item()
    return seq, final_dist


# =====================================================================
# Protocol 1 — Recall@K from ground-truth val trajectories
# =====================================================================

def evaluate_protocol_1(
    model: JEPAWorldModel,
    sampler: HDF5BatchSampler,
    norm_stats: dict,
    device: torch.device,
    horizons: List[int] = (5, 10, 25),
    n_pairs_per_h: int = 50,
    cem_n_samples: int = 200,
    cem_n_iter: int = 5,
) -> Dict:
    """For each horizon h, sample (s_t, a_gt[t..t+h-1], s_{t+h}) from val.

    The ground-truth action sequence a_gt is by construction a "good plan":
    it actually leads from s_t to s_{t+h} in the dataset. We compare:

      d_gt   = ||predict(z_t, a_gt) - z_g||²    (latent dist of GT plan)
      d_cem  = ||predict(z_t, a_cem) - z_g||²   (latent dist of CEM plan)
      d_rand = ||predict(z_t, a_rand) - z_g||²  (latent dist of random)

    Question: does CEM find a plan as good as GT, or better?
    """
    rng = np.random.default_rng(2)
    results = {}

    for h in horizons:
        d_gt_list   = []
        d_cem_list  = []
        d_rand_list = []
        cem_better_than_gt = 0
        cem_within_10pct_of_gt = 0
        cem_t_total = 0.0

        for i in range(n_pairs_per_h):
            # Sample one (s_t, a_gt, s_g) tuple from val
            raw = sampler.sample_batch(1, h, rng)
            z_t, z_g = encode_pair(model, raw['s_t'], raw['s_tk'],
                                    norm_stats, device)
            z_t = z_t.squeeze(0)
            z_g = z_g.squeeze(0)
            a_gt = torch.from_numpy(raw['actions'][0]).float().to(device)  # [h, 8]

            # Distance of ground-truth plan
            d_gt = score_sequences(model, z_t, a_gt.unsqueeze(0), z_g)[0].item()
            d_gt_list.append(d_gt)

            # CEM plan
            t0 = time.time()
            _, d_cem = cem_plan(
                model, z_t, z_g, h,
                n_samples=cem_n_samples, n_iter=cem_n_iter,
                device=device,
            )
            cem_t_total += time.time() - t0
            d_cem_list.append(d_cem)

            # Random baseline
            _, d_rand = random_baseline(
                model, z_t, z_g, h, n_samples=cem_n_samples,
            )
            d_rand_list.append(d_rand)

            if d_cem < d_gt:
                cem_better_than_gt += 1
            if d_cem < d_gt * 1.1:
                cem_within_10pct_of_gt += 1

        d_gt_arr   = np.array(d_gt_list)
        d_cem_arr  = np.array(d_cem_list)
        d_rand_arr = np.array(d_rand_list)

        results[h] = {
            'n_pairs': n_pairs_per_h,
            'd_gt_mean':   float(d_gt_arr.mean()),
            'd_cem_mean':  float(d_cem_arr.mean()),
            'd_rand_mean': float(d_rand_arr.mean()),
            'd_gt_median':   float(np.median(d_gt_arr)),
            'd_cem_median':  float(np.median(d_cem_arr)),
            'pct_cem_better_than_gt':       100 * cem_better_than_gt / n_pairs_per_h,
            'pct_cem_within_10pct_of_gt':   100 * cem_within_10pct_of_gt / n_pairs_per_h,
            'cem_avg_time_ms': 1000 * cem_t_total / n_pairs_per_h,
        }

    return results


# =====================================================================
# Protocol 3 — CEM vs Random vs Greedy
# =====================================================================

def evaluate_protocol_3(
    model: JEPAWorldModel,
    sampler: HDF5BatchSampler,
    norm_stats: dict,
    device: torch.device,
    horizons: List[int] = (5, 10, 25),
    n_pairs_per_h: int = 100,
    cem_n_samples: int = 200,
    cem_n_iter: int = 5,
) -> Dict:
    """For each horizon h, compare CEM / Random / Greedy on goal-reaching."""
    rng = np.random.default_rng(3)
    results = {}

    for h in horizons:
        d_cem_list = []
        d_rand_list = []
        d_greedy_list = []
        t_cem_list = []
        t_rand_list = []
        t_greedy_list = []

        for i in range(n_pairs_per_h):
            raw = sampler.sample_batch(1, h, rng)
            z_t, z_g = encode_pair(model, raw['s_t'], raw['s_tk'],
                                    norm_stats, device)
            z_t = z_t.squeeze(0)
            z_g = z_g.squeeze(0)

            t0 = time.time()
            _, d_cem = cem_plan(
                model, z_t, z_g, h,
                n_samples=cem_n_samples, n_iter=cem_n_iter, device=device,
            )
            t_cem_list.append(time.time() - t0)
            d_cem_list.append(d_cem)

            t0 = time.time()
            _, d_rand = random_baseline(
                model, z_t, z_g, h, n_samples=cem_n_samples,
            )
            t_rand_list.append(time.time() - t0)
            d_rand_list.append(d_rand)

            t0 = time.time()
            _, d_greedy = greedy_baseline(model, z_t, z_g, h, n_candidates=100)
            t_greedy_list.append(time.time() - t0)
            d_greedy_list.append(d_greedy)

        d_cem    = np.array(d_cem_list)
        d_rand   = np.array(d_rand_list)
        d_greedy = np.array(d_greedy_list)

        # Win rates
        cem_beats_rand    = int((d_cem < d_rand).sum())
        cem_beats_greedy  = int((d_cem < d_greedy).sum())

        results[h] = {
            'n_pairs': n_pairs_per_h,
            'cem_mean':    float(d_cem.mean()),
            'random_mean': float(d_rand.mean()),
            'greedy_mean': float(d_greedy.mean()),
            'cem_median':    float(np.median(d_cem)),
            'random_median': float(np.median(d_rand)),
            'greedy_median': float(np.median(d_greedy)),
            'pct_cem_beats_random':  100 * cem_beats_rand   / n_pairs_per_h,
            'pct_cem_beats_greedy':  100 * cem_beats_greedy / n_pairs_per_h,
            'cem_time_ms':    1000 * float(np.mean(t_cem_list)),
            'random_time_ms': 1000 * float(np.mean(t_rand_list)),
            'greedy_time_ms': 1000 * float(np.mean(t_greedy_list)),
        }

    return results


# =====================================================================
# Pretty printing
# =====================================================================

def print_protocol_1(results: Dict):
    print("\n" + "=" * 70)
    print("Protocol 1 — CEM vs ground-truth plan from val trajectories")
    print("=" * 70)
    print(f"  {'h':>4s}  {'d_gt':>9s}  {'d_cem':>9s}  {'d_rand':>9s}  "
          f"{'%≤GT':>6s}  {'%<GT':>6s}  {'CEM (ms)':>9s}")
    print("  " + "-" * 64)
    for h, r in results.items():
        print(f"  {h:>4d}  "
              f"{r['d_gt_mean']:>9.4f}  "
              f"{r['d_cem_mean']:>9.4f}  "
              f"{r['d_rand_mean']:>9.4f}  "
              f"{r['pct_cem_within_10pct_of_gt']:>5.1f}%  "
              f"{r['pct_cem_better_than_gt']:>5.1f}%  "
              f"{r['cem_avg_time_ms']:>9.1f}")
    print()
    print("  Reading:")
    print("    d_gt:   latent distance of ground-truth action sequence")
    print("    d_cem:  latent distance of CEM-found sequence")
    print("    d_rand: latent distance of best-of-200 random sequences")
    print("    %≤GT:   % of pairs where CEM gets within 10% of GT distance")
    print("    %<GT:   % of pairs where CEM beats GT (uses WM imperfection)")
    print()
    print("  Good signs:")
    print("    - d_cem ≪ d_rand   → CEM learned to use the WM")
    print("    - d_cem ≈ d_gt     → CEM finds plans as good as ground truth")


def print_protocol_3(results: Dict):
    print("\n" + "=" * 70)
    print("Protocol 3 — CEM vs Random vs Greedy")
    print("=" * 70)
    print(f"  {'h':>4s}  {'CEM':>9s}  {'Random':>9s}  {'Greedy':>9s}  "
          f"{'CEM>R%':>7s}  {'CEM>G%':>7s}  {'CEM ms':>7s}")
    print("  " + "-" * 66)
    for h, r in results.items():
        print(f"  {h:>4d}  "
              f"{r['cem_mean']:>9.4f}  "
              f"{r['random_mean']:>9.4f}  "
              f"{r['greedy_mean']:>9.4f}  "
              f"{r['pct_cem_beats_random']:>6.1f}%  "
              f"{r['pct_cem_beats_greedy']:>6.1f}%  "
              f"{r['cem_time_ms']:>7.1f}")
    print()
    print("  Reading: lower distance = better plan in latent space")
    print()
    print("  Good signs:")
    print("    - CEM mean < Greedy mean < Random mean (sane ordering)")
    print("    - CEM>R% > 80%   → CEM consistently beats random")
    print("    - CEM>G% > 60%   → CEM beats greedy myopic search")


# =====================================================================
# Main
# =====================================================================

def evaluate_planning(
    checkpoint_path: str,
    dataset_path: str,
    norm_stats_path: str,
    output_dir: str,
    device_str: str = 'auto',
    n_pairs_p1: int = 50,
    n_pairs_p3: int = 100,
    cem_n_samples: int = 200,
    cem_n_iter: int = 5,
):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    print("=" * 70)
    print("JEPA Planning Evaluation")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device}")
    print(f"  CEM:        N={cem_n_samples}, n_iter={cem_n_iter}")
    print(f"  Pairs P1:   {n_pairs_p1} per horizon")
    print(f"  Pairs P3:   {n_pairs_p3} per horizon")

    model = load_model(checkpoint_path, device)
    norm_stats = load_norm_stats(norm_stats_path)
    sampler = HDF5BatchSampler(dataset_path, split='val')

    # Protocol 1
    print("\n>>> Running Protocol 1 (Recall vs ground-truth plans)...")
    t0 = time.time()
    p1 = evaluate_protocol_1(
        model, sampler, norm_stats, device,
        horizons=[5, 10],
        n_pairs_per_h=n_pairs_p1,
        cem_n_samples=cem_n_samples, cem_n_iter=cem_n_iter,
    )
    print(f"  Protocol 1 took {time.time() - t0:.1f}s")

    # Protocol 3
    print("\n>>> Running Protocol 3 (CEM vs Random vs Greedy)...")
    t0 = time.time()
    p3 = evaluate_protocol_3(
        model, sampler, norm_stats, device,
        horizons=[5, 10, 25],
        n_pairs_per_h=n_pairs_p3,
        cem_n_samples=cem_n_samples, cem_n_iter=cem_n_iter,
    )
    print(f"  Protocol 3 took {time.time() - t0:.1f}s")

    print_protocol_1(p1)
    print_protocol_3(p3)

    # Save
    out_path = Path(output_dir) / 'planning_results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'checkpoint': checkpoint_path,
            'protocol_1': {str(h): v for h, v in p1.items()},
            'protocol_3': {str(h): v for h, v in p3.items()},
            'cem_config': {
                'n_samples': cem_n_samples,
                'n_iter': cem_n_iter,
            },
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    sampler.close()
    return {'protocol_1': p1, 'protocol_3': p3}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--dataset',     default='data/optical_wm_poc.h5')
    p.add_argument('--norm-stats',  default='data/norm_stats.json')
    p.add_argument('--output',      default='checkpoints')
    p.add_argument('--device',      default='auto')
    p.add_argument('--pairs-p1',    type=int, default=50)
    p.add_argument('--pairs-p3',    type=int, default=100)
    p.add_argument('--cem-samples', type=int, default=200)
    p.add_argument('--cem-iter',    type=int, default=5)
    args = p.parse_args()

    evaluate_planning(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        norm_stats_path=args.norm_stats,
        output_dir=args.output,
        device_str=args.device,
        n_pairs_p1=args.pairs_p1,
        n_pairs_p3=args.pairs_p3,
        cem_n_samples=args.cem_samples,
        cem_n_iter=args.cem_iter,
    )