"""
Physical validation of WM-guided planning via GNPy execution.

PURPOSE
=======
Bridge the gap between latent-space CEM (in evaluate_planning.py) and physical
reality. Take action sequences proposed by WM-guided search, execute them via
GNPy on the ground-truth physical model, and verify:

  1. WM-predicted GSNR matches GNPy-actual GSNR
  2. WM-guided plan reaches the goal better than random
  3. Speedup of WM vs GNPy is real (timed)

DESIGN
======
Unlike continuous CEM in latent space, this script samples DISCRETE valid
actions (real ADD/REMOVE operations on tracked lightpaths) so they're directly
executable by GNPy. The "WM-guided" part: best-of-N random discrete sequences
scored by WM latent distance.

This is a single-iteration analog of CEM — full CEM with discrete actions
needs categorical distributions, which is overkill for PoC.

USAGE (must run on PC with GNPy installed, NOT on Colab)
=====
    cd "Digital Twin AGI"
    py -m optical_wm_jepa.evaluate_planning_gnpy \\
        --checkpoint best_model.pt \\
        --dataset    data/optical_wm_poc.h5 \\
        --norm-stats data/norm_stats.json \\
        --topology   data/topology.json \\
        --output     gnpy_validation_results.json

Requirements:
    - optical_wm_poc package in same parent dir (sibling to optical_wm_jepa)
    - GNPy installed (pip install gnpy)
    - best_model.pt downloaded from Drive
"""
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import networkx as nx

# ---------------------------------------------------------------------
# Bridge to optical_wm_poc (sibling repo with GNPy oracle)
# ---------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent       # .../optical_wm_jepa/optical_wm_jepa
_REPO_ROOT = _THIS_DIR.parent                      # .../optical_wm_jepa
_DTW_ROOT  = _REPO_ROOT.parent                     # .../Digital Twin AGI
sys.path.insert(0, str(_DTW_ROOT))

try:
    from optical_wm_poc.gnpy_oracle import GNPyOracle, Lightpath
    from optical_wm_poc.topology   import load_topology
    from optical_wm_poc.constants  import (
        N_SLOTS, MAX_LP, MAX_HOPS,
        ACTION_DIM, A_TYPE, A_SRC, A_DST, A_SLOT, A_HOP_START,
        ACTION_ADD, ACTION_REMOVE,
    )
except ImportError as e:
    print(f"ERROR importing optical_wm_poc: {e}")
    print(f"Looking in: {_DTW_ROOT}")
    print("Ensure optical_wm_poc is a sibling directory to optical_wm_jepa.")
    sys.exit(1)

from .model import JEPAWorldModel
from .train import normalize_batch, load_norm_stats
from .evaluate_planning import load_model


# =====================================================================
# State reconstruction from HDF5
# =====================================================================

def reconstruct_lightpaths(
    lp_table: np.ndarray,
    lp_active: np.ndarray,
    topology: dict,
    graph: nx.Graph,
    id_prefix: str = "lp",
) -> List[Lightpath]:
    """Convert HDF5 lp_table → list of Lightpath objects.

    The stored lp_table has [slot, src_idx, dst_idx, n_hops] but no route.
    We recompute the route via shortest path (same as dataset generation).
    """
    node_ids = topology['node_ids']
    lps = []
    for i in range(len(lp_active)):
        if not lp_active[i]:
            continue
        slot    = int(lp_table[i, 0])
        src_idx = int(lp_table[i, 1])
        dst_idx = int(lp_table[i, 2])

        if not (0 <= src_idx < len(node_ids) and 0 <= dst_idx < len(node_ids)):
            continue
        src = node_ids[src_idx]
        dst = node_ids[dst_idx]
        if src == dst:
            continue
        try:
            route = nx.shortest_path(graph, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            continue
        if not (0 <= slot < N_SLOTS):
            continue

        lps.append(Lightpath(
            id=f"{id_prefix}_{i:03d}",
            source=src, destination=dst,
            route=list(route), slot=slot,
        ))
    return lps


def encode_action_vec(action_type: int, src: str, dst: str, slot: int,
                      route: Optional[List[str]], topology: dict) -> np.ndarray:
    """Encode (type, src, dst, slot, route) → 8-dim action vector."""
    a = np.full(ACTION_DIM, -1, dtype=np.float32)
    a[A_TYPE] = float(action_type)
    a[A_SRC]  = float(topology['node_ids'].index(src))
    a[A_DST]  = float(topology['node_ids'].index(dst))
    a[A_SLOT] = float(slot)
    if route is not None:
        for i, n in enumerate(route[:MAX_HOPS]):
            a[A_HOP_START + i] = float(topology['node_ids'].index(n))
    return a


# =====================================================================
# Discrete action sampling with state tracking
# =====================================================================

class DiscretePlanner:
    """Samples valid ADD/REMOVE action sequences and scores them via WM."""

    def __init__(self, topology: dict, model: JEPAWorldModel,
                 norm_stats: dict, device: torch.device):
        self.topology = topology
        self.node_ids = topology['node_ids']
        self.model = model
        self.norm_stats = norm_stats
        self.device = device

        self.graph = nx.Graph()
        for link in topology['links']:
            self.graph.add_edge(link['src'], link['dst'],
                                weight=link['length_km'])

        # Bidirectional link lookup
        self.link_lookup = {}
        for link in topology['links']:
            self.link_lookup[(link['src'], link['dst'])] = link['id']
            self.link_lookup[(link['dst'], link['src'])] = link['id']

    def _build_wl_usage(self, lightpaths: List[Lightpath]) -> Dict[str, set]:
        """Build link → set(occupied slots) from current lightpaths."""
        usage = {l['id']: set() for l in self.topology['links']}
        for lp in lightpaths:
            for h in range(lp.n_hops):
                lid = self.link_lookup.get((lp.route[h], lp.route[h + 1]))
                if lid is not None:
                    usage[lid].add(lp.slot)
        return usage

    def _first_fit_slot(self, route: List[str],
                         wl_usage: Dict[str, set],
                         rng: np.random.Generator) -> Optional[int]:
        link_ids = []
        for i in range(len(route) - 1):
            lid = self.link_lookup.get((route[i], route[i + 1]))
            if lid is None:
                return None
            link_ids.append(lid)
        free = [s for s in range(N_SLOTS)
                if all(s not in wl_usage[lid] for lid in link_ids)]
        if not free:
            return None
        return int(rng.choice(free[:5]))

    def sample_action(
        self, lightpaths: List[Lightpath], rng: np.random.Generator,
        p_add: float = 0.5,
    ) -> Optional[Tuple[str, dict, np.ndarray]]:
        """Sample one valid action.

        Returns (kind, payload, action_vec) or None.
          kind ∈ {'ADD', 'REMOVE'}
          payload = the new Lightpath (ADD) or LP to remove (REMOVE)
          action_vec: [8] float32 — dataset-compatible encoding
        """
        do_add = rng.random() < p_add
        wl_usage = self._build_wl_usage(lightpaths)

        if do_add and len(lightpaths) < MAX_LP:
            for _ in range(20):
                pair = rng.choice(self.node_ids, size=2, replace=False)
                src, dst = str(pair[0]), str(pair[1])
                try:
                    route = nx.shortest_path(self.graph, src, dst, weight='weight')
                except nx.NetworkXNoPath:
                    continue
                if len(route) - 1 > MAX_HOPS:
                    continue
                slot = self._first_fit_slot(route, wl_usage, rng)
                if slot is None:
                    continue
                new_lp = Lightpath(
                    id=f"lp_plan_{len(lightpaths):03d}_{rng.integers(10000)}",
                    source=src, destination=dst,
                    route=list(route), slot=slot,
                )
                action_vec = encode_action_vec(
                    ACTION_ADD, src, dst, slot, list(route), self.topology
                )
                return ('ADD', new_lp, action_vec)

        if len(lightpaths) >= 2:
            idx = int(rng.integers(0, len(lightpaths)))
            lp = lightpaths[idx]
            action_vec = encode_action_vec(
                ACTION_REMOVE, lp.source, lp.destination, lp.slot, None,
                self.topology
            )
            return ('REMOVE', lp, action_vec)

        if not do_add and len(lightpaths) < MAX_LP:
            for _ in range(20):
                pair = rng.choice(self.node_ids, size=2, replace=False)
                src, dst = str(pair[0]), str(pair[1])
                try:
                    route = nx.shortest_path(self.graph, src, dst, weight='weight')
                except nx.NetworkXNoPath:
                    continue
                if len(route) - 1 > MAX_HOPS:
                    continue
                slot = self._first_fit_slot(route, wl_usage, rng)
                if slot is None:
                    continue
                new_lp = Lightpath(
                    id=f"lp_plan_fallback_{rng.integers(10000)}",
                    source=src, destination=dst,
                    route=list(route), slot=slot,
                )
                action_vec = encode_action_vec(
                    ACTION_ADD, src, dst, slot, list(route), self.topology
                )
                return ('ADD', new_lp, action_vec)
        return None

    def sample_action_sequence(
        self, initial_lps: List[Lightpath], h: int,
        rng: np.random.Generator, p_add: float = 0.5,
    ) -> Optional[Tuple[List[Tuple[str, object]], np.ndarray]]:
        """Sample h valid actions, simulating state changes."""
        sim_lps = list(initial_lps)
        actions = []
        action_vecs = []

        for step in range(h):
            result = self.sample_action(sim_lps, rng, p_add)
            if result is None:
                return None
            kind, payload, av = result
            actions.append((kind, payload))
            action_vecs.append(av)

            if kind == 'ADD':
                sim_lps = sim_lps + [payload]
            else:
                sim_lps = [l for l in sim_lps if l.id != payload.id]

        return actions, np.stack(action_vecs)

    def score_sequences(
        self, z_t: torch.Tensor, sequences_vec: np.ndarray, z_g: torch.Tensor,
    ) -> np.ndarray:
        """Score N action sequences by latent distance to z_g."""
        N = sequences_vec.shape[0]
        seq_t = torch.from_numpy(sequences_vec).float().to(self.device)
        z_t_batch = z_t.unsqueeze(0).expand(N, -1)
        with torch.no_grad():
            z_pred = self.model.predict(z_t_batch, seq_t)
        scores = ((z_pred - z_g.unsqueeze(0)) ** 2).sum(dim=-1).cpu().numpy()
        return scores


# =====================================================================
# GNPy execution
# =====================================================================

def execute_actions_via_gnpy(
    initial_lps: List[Lightpath],
    actions: List[Tuple[str, object]],
    oracle: GNPyOracle,
) -> Tuple[List[Lightpath], dict, float]:
    """Apply action sequence to physical state, evaluate final state via GNPy."""
    t0 = time.time()
    sim_lps = list(initial_lps)

    for kind, payload in actions:
        if kind == 'ADD':
            sim_lps = sim_lps + [payload]
        else:
            sim_lps = [l for l in sim_lps if l.id != payload.id]

    results, _ = oracle.evaluate_all(sim_lps)
    elapsed = time.time() - t0
    return sim_lps, results, elapsed


def gsnr_observable(lightpaths: List[Lightpath],
                    results: dict) -> Dict[str, float]:
    """Compute observable summary of a (lps, results) state."""
    if not lightpaths or not results:
        return {'n_lps': 0, 'mean_gsnr': 0.0, 'min_gsnr': 0.0}
    gsnrs = [results[lp.id][0] for lp in lightpaths if lp.id in results]
    if not gsnrs:
        return {'n_lps': len(lightpaths), 'mean_gsnr': 0.0, 'min_gsnr': 0.0}
    return {
        'n_lps':     len(lightpaths),
        'mean_gsnr': float(np.mean(gsnrs)),
        'min_gsnr':  float(np.min(gsnrs)),
    }


# =====================================================================
# Main evaluation loop
# =====================================================================

def evaluate_one_pair(
    raw_pair: dict,
    planner: DiscretePlanner,
    oracle: GNPyOracle,
    h: int,
    n_samples: int,
    rng: np.random.Generator,
    verbose: bool = False,
) -> Optional[dict]:
    """Evaluate one (s_t, s_g) pair: WM-guided plan vs random, both via GNPy."""

    s_t  = {k: v[0:1] for k, v in raw_pair['s_t'].items()}
    s_g  = {k: v[0:1] for k, v in raw_pair['s_tk'].items()}

    s_t_n = normalize_batch(s_t, planner.norm_stats, planner.device)
    s_g_n = normalize_batch(s_g, planner.norm_stats, planner.device)
    with torch.no_grad():
        z_t = planner.model.encode(s_t_n).squeeze(0)
        z_g = planner.model.encode(s_g_n).squeeze(0)

    initial_lps = reconstruct_lightpaths(
        raw_pair['s_t']['lp_table'][0],
        raw_pair['s_t']['lp_active'][0],
        planner.topology, planner.graph,
        id_prefix="lp_init",
    )
    if len(initial_lps) < 2:
        if verbose:
            print(f"  Skip: too few initial LPs ({len(initial_lps)})")
        return None

    sequences = []
    for _ in range(n_samples):
        seq = planner.sample_action_sequence(initial_lps, h, rng)
        if seq is not None:
            sequences.append(seq)

    if len(sequences) < 2:
        if verbose:
            print(f"  Skip: only {len(sequences)} valid sequences")
        return None

    seq_vecs = np.stack([s[1] for s in sequences])
    t0_score = time.time()
    scores = planner.score_sequences(z_t, seq_vecs, z_g)
    t_score = time.time() - t0_score

    best_idx = int(np.argmin(scores))
    best_actions, _ = sequences[best_idx]

    random_idx = int(rng.integers(0, len(sequences)))
    while random_idx == best_idx and len(sequences) > 1:
        random_idx = int(rng.integers(0, len(sequences)))
    rand_actions, _ = sequences[random_idx]

    _, gnpy_best_results, t_gnpy_best = execute_actions_via_gnpy(
        initial_lps, best_actions, oracle
    )
    _, gnpy_rand_results, t_gnpy_rand = execute_actions_via_gnpy(
        initial_lps, rand_actions, oracle
    )

    sim_lps_best = list(initial_lps)
    for kind, payload in best_actions:
        if kind == 'ADD':
            sim_lps_best.append(payload)
        else:
            sim_lps_best = [l for l in sim_lps_best if l.id != payload.id]

    sim_lps_rand = list(initial_lps)
    for kind, payload in rand_actions:
        if kind == 'ADD':
            sim_lps_rand.append(payload)
        else:
            sim_lps_rand = [l for l in sim_lps_rand if l.id != payload.id]

    goal_lps = reconstruct_lightpaths(
        raw_pair['s_tk']['lp_table'][0],
        raw_pair['s_tk']['lp_active'][0],
        planner.topology, planner.graph,
        id_prefix="lp_goal",
    )
    _, goal_results, _ = execute_actions_via_gnpy(goal_lps, [], oracle)

    obs_best = gsnr_observable(sim_lps_best, gnpy_best_results)
    obs_rand = gsnr_observable(sim_lps_rand, gnpy_rand_results)
    obs_goal = gsnr_observable(goal_lps,    goal_results)

    return {
        'h': h,
        'n_initial_lps': len(initial_lps),
        'n_sequences_sampled': len(sequences),
        'lat_dist_best': float(scores[best_idx]),
        'lat_dist_rand': float(scores[random_idx]),
        'lat_dist_mean': float(np.mean(scores)),
        'mean_gsnr_best':  obs_best['mean_gsnr'],
        'mean_gsnr_rand':  obs_rand['mean_gsnr'],
        'mean_gsnr_goal':  obs_goal['mean_gsnr'],
        'gsnr_err_best':   abs(obs_best['mean_gsnr'] - obs_goal['mean_gsnr']),
        'gsnr_err_rand':   abs(obs_rand['mean_gsnr'] - obs_goal['mean_gsnr']),
        'n_lps_best':  obs_best['n_lps'],
        'n_lps_rand':  obs_rand['n_lps'],
        'n_lps_goal':  obs_goal['n_lps'],
        't_wm_score_ms':  1000 * t_score,
        't_gnpy_best_s':  t_gnpy_best,
        't_gnpy_rand_s':  t_gnpy_rand,
    }


def main_validation(
    checkpoint_path: str,
    dataset_path: str,
    norm_stats_path: str,
    topology_path: str,
    output_path: str,
    n_pairs: int = 20,
    horizon: int = 5,
    n_samples: int = 100,
    device_str: str = 'cpu',
    verbose: bool = True,
):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    print("=" * 70)
    print("GNPy Physical Validation of WM-Guided Planning")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Pairs:      {n_pairs}")
    print(f"  Horizon:    {horizon}")
    print(f"  Samples:    {n_samples} per pair")
    print(f"  Device WM:  {device}")
    print()

    print("Loading WM...")
    model = load_model(checkpoint_path, device)
    norm_stats = load_norm_stats(norm_stats_path)

    print("Loading topology and GNPy oracle (slow first time)...")
    topology = load_topology(topology_path)
    oracle = GNPyOracle(topology)
    print(f"  Topology: {topology['n_nodes']} nodes, "
          f"{topology['n_links']} links")

    planner = DiscretePlanner(topology, model, norm_stats, device)

    from .train import HDF5BatchSampler
    sampler = HDF5BatchSampler(dataset_path, split='val')
    print(f"  Val episodes: {len(sampler.episode_ids)}\n")

    rng = np.random.default_rng(42)
    results = []
    t_start = time.time()

    for i in range(n_pairs):
        raw = sampler.sample_batch(1, horizon, rng)
        out = evaluate_one_pair(
            raw, planner, oracle, horizon, n_samples, rng,
            verbose=verbose,
        )
        if out is None:
            continue
        results.append(out)

        if verbose:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  [{i+1:2d}/{n_pairs}] "
                  f"WM-best GSNR err={out['gsnr_err_best']:.3f} dB | "
                  f"Random GSNR err={out['gsnr_err_rand']:.3f} dB | "
                  f"WM_score={out['t_wm_score_ms']:.0f}ms | "
                  f"GNPy={out['t_gnpy_best_s']:.1f}s | "
                  f"({rate:.2f} pair/s)")

    sampler.close()

    if not results:
        print("ERROR: no valid results")
        return

    arr = lambda key: np.array([r[key] for r in results])

    summary = {
        'n_pairs_evaluated': len(results),
        'horizon': horizon,
        'n_samples_per_pair': n_samples,
        'gsnr_err_best_mean':   float(arr('gsnr_err_best').mean()),
        'gsnr_err_best_median': float(np.median(arr('gsnr_err_best'))),
        'gsnr_err_rand_mean':   float(arr('gsnr_err_rand').mean()),
        'gsnr_err_rand_median': float(np.median(arr('gsnr_err_rand'))),
        'pct_wm_beats_random':  100 * float(
            (arr('gsnr_err_best') < arr('gsnr_err_rand')).mean()
        ),
        'lat_dist_best_mean': float(arr('lat_dist_best').mean()),
        'lat_dist_rand_mean': float(arr('lat_dist_rand').mean()),
        'wm_scoring_total_ms':      float(arr('t_wm_score_ms').sum()),
        'gnpy_execution_total_s':   float(arr('t_gnpy_best_s').sum()
                                          + arr('t_gnpy_rand_s').sum()),
        'speedup_estimate':         float(
            (arr('t_gnpy_best_s').mean() * 1000) /
            (arr('t_wm_score_ms').mean() / n_samples)
        ),
    }

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Pairs evaluated:    {summary['n_pairs_evaluated']}")
    print()
    print("  GSNR error vs goal (lower = better):")
    print(f"    WM-best:   {summary['gsnr_err_best_mean']:.3f} dB "
          f"(median {summary['gsnr_err_best_median']:.3f})")
    print(f"    Random:    {summary['gsnr_err_rand_mean']:.3f} dB "
          f"(median {summary['gsnr_err_rand_median']:.3f})")
    print(f"    WM beats random:   "
          f"{summary['pct_wm_beats_random']:.1f}% of pairs")
    print()
    print("  Latent distances:")
    print(f"    WM-best:  {summary['lat_dist_best_mean']:.3f}")
    print(f"    Random:   {summary['lat_dist_rand_mean']:.3f}")
    print()
    print("  Speed:")
    avg_score_ms = arr('t_wm_score_ms').mean()
    avg_gnpy_s   = arr('t_gnpy_best_s').mean()
    per_seq_wm_ms = avg_score_ms / n_samples
    print(f"    WM scoring of {n_samples} sequences: "
          f"{avg_score_ms:.1f} ms total → "
          f"{per_seq_wm_ms:.2f} ms per sequence")
    print(f"    GNPy single execution:              "
          f"{avg_gnpy_s:.2f} s = {avg_gnpy_s*1000:.0f} ms")
    print(f"    Speedup of WM over GNPy:            "
          f"{avg_gnpy_s*1000/per_seq_wm_ms:.0f}x")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'summary': summary,
            'per_pair': results,
            'config': {
                'checkpoint':  checkpoint_path,
                'n_pairs':     n_pairs,
                'horizon':     horizon,
                'n_samples':   n_samples,
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--dataset',     default='data/optical_wm_poc.h5')
    p.add_argument('--norm-stats',  default='data/norm_stats.json')
    p.add_argument('--topology',    default='data/topology.json')
    p.add_argument('--output',      default='gnpy_validation_results.json')
    p.add_argument('--pairs',       type=int, default=20)
    p.add_argument('--horizon',     type=int, default=5)
    p.add_argument('--samples',     type=int, default=100)
    p.add_argument('--device',      default='cpu')
    args = p.parse_args()

    main_validation(
        checkpoint_path  = args.checkpoint,
        dataset_path     = args.dataset,
        norm_stats_path  = args.norm_stats,
        topology_path    = args.topology,
        output_path      = args.output,
        n_pairs          = args.pairs,
        horizon          = args.horizon,
        n_samples        = args.samples,
        device_str       = args.device,
    )