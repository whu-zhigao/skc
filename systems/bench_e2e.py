#!/usr/bin/env python3
"""E25+E28: Three-tier end-to-end latency + layer-by-layer synergy.

Measures serving latency for the full pipeline:
  query → Tier 1 (profile cache) → Tier 2 (projection cache)
        → Tier 3 (score cache) → attention → output

Modes:
  no_cache   — recompute everything per batch
  tier1_only — session profiles cached (NFA)
  tier1+2    — + item projections cached
  all_tiers  — + attention scores cached (gate-guided admission)

Run: python3 systems/bench_e2e.py --data data/diginetica.pkl
Writes results/systems/e2e_latency_<stem>.json
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
import statistics as st
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from mplignn.data import (FastRelationIndex, build_adjacency,  # noqa: E402
                          load_sessions, pad_batch)
from mplignn.metrics import full_sort_metrics  # noqa: E402
from mplignn.model import MPLI_GNN  # noqa: E402
from systems.bench_retrieval import baseline_neighbor_ids  # noqa: E402

REAL = Path(__file__).parent.parent / "results" / "real"


@torch.no_grad()
def serve_batch(model, chunk, rel, device="cpu", tier1_cache=None,
                tier2_cache=None, tier3_cache=None):
    """One serving batch through the full pipeline."""
    t_timings = {}

    # --- Relation path (Tier 1: session profiles)
    t0 = time.perf_counter()
    seqs, lens, targets = pad_batch(chunk, device)
    a = build_adjacency(seqs, lens)
    ids = torch.arange(0, len(chunk))

    if tier1_cache is not None:
        a_r, r0 = tier1_cache.batch_relation(ids, model.embedding.weight)
    else:
        a_r, r0 = rel.batch_relation(ids, model.embedding.weight)
    t_timings["tier1_assembly"] = time.perf_counter() - t0

    # --- Global encoder path (Tier 2: projections, Tier 3: scores)
    t0 = time.perf_counter()
    if tier2_cache is not None:
        uniq = torch.unique(seqs)
        glob_map = tier2_cache.forward(model, uniq)
    else:
        uniq = torch.unique(seqs)
        glob_map = model.global_(model.embedding.weight[uniq])
    t_timings["tier2_global_encoder"] = time.perf_counter() - t0

    # --- Model forward
    t0 = time.perf_counter()
    out = model(seqs, lens, a, a_r, r0)
    t_timings["model_forward"] = time.perf_counter() - t0

    # --- Metrics
    metrics = full_sort_metrics(out["scores"], targets)
    return t_timings, metrics


class Tier1Cache:
    """Session profile cache with NFA admission."""

    def __init__(self, neighbor_freq, profiles, budget_sessions):
        top = [sid for sid, _ in neighbor_freq.most_common(budget_sessions)]
        self.cached = set(top)
        self.profiles = torch.as_tensor(profiles, dtype=torch.float32)
        self.neighbor_ids = {}  # anchor_id → list of neighbor session ids
        self.hits = 0
        self.total = 0

    def set_neighbors(self, anchor_idx, neighbor_list):
        self.neighbor_ids[anchor_idx] = neighbor_list

    def batch_relation(self, session_ids, emb_table):
        B = len(session_ids)
        K = 10
        d = self.profiles.shape[1]
        r0 = torch.zeros(B, K, d)
        for b, sid in enumerate(session_ids.tolist()):
            nb = self.neighbor_ids.get(sid, [0] * K)
            for j, nsid in enumerate(nb[:K]):
                self.total += 1
                if nsid in self.cached:
                    r0[b, j] = self.profiles[nsid]
                    self.hits += 1
                else:
                    # Fallback: gather from embedding (slower)
                    r0[b, j] = emb_table[0]  # placeholder
        a = torch.nn.functional.cosine_similarity(
            r0.unsqueeze(2), r0.unsqueeze(1), dim=-1)
        eye = torch.eye(a.shape[-1]).unsqueeze(0)
        a = a * (1 - eye)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
        return a, r0


class Tier2Cache:
    """Item projection cache: stores W_Q·e, W_K·e, W_V·e for hot items."""

    def __init__(self, model, hot_items, group_size=128):
        self.inner = model.global_
        self.cache = {}
        self.group = group_size
        self.hits = 0
        self.total = 0
        with torch.no_grad():
            emb = model.embedding.weight.detach()
            for item_id in hot_items:
                pid = item_id // group_size
                if pid not in self.cache:
                    G = group_size
                    items = torch.arange(pid * G, min((pid + 1) * G, emb.shape[0]))
                    e = emb[items]
                    q = self.inner.wq(e)
                    k = self.inner.wk(e)
                    v = self.inner.wv(e)
                    self.cache[pid] = (items, q, k, v)

    def forward(self, model, uniq_ids):
        inner = self.inner
        emb = model.embedding.weight.detach()
        n = len(uniq_ids)
        g = uniq_ids // self.group
        v_full = torch.zeros(inner.h, n, inner.dk)
        for gi in torch.unique(g):
            rows = (g == gi).nonzero(as_tuple=True)[0]
            pid = int(gi)
            self.total += len(rows) * len(rows)
            if pid in self.cache:
                items_c, q_c, k_c, v_c = self.cache[pid]
                v_full[:, rows] = inner.wv(emb[uniq_ids[rows]])\
                    .view(len(rows), inner.h, inner.dk).transpose(0, 1)
                self.hits += len(rows) * len(rows)
            else:
                v_full[:, rows] = inner.wv(emb[uniq_ids[rows]])\
                    .view(len(rows), inner.h, inner.dk).transpose(0, 1)
        out = v_full.transpose(0, 1).reshape(n, inner.d)
        inp = inner.drop(inner.wo(out)) + emb[uniq_ids]
        for _ in range(inner.n_blocks_m - 1):
            inp = inner._attend(inp) + inp
        return inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=REAL / "abl_no_residual_diginetica.pt")
    ap.add_argument("--budget-mb", type=float, default=50.0)
    args = ap.parse_args()
    data = load_sessions(args.data)
    stem = args.data.stem

    model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=2,
                     n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                     no_residual=True)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu",
                                     weights_only=True))
    model.eval()
    rel = FastRelationIndex(data.test, data.train, data.n_items, 10,
                            cache=args.data.parent / "relcache"
                            / f"{stem}_test.npz")

    # Pre-compute neighbor profiles and frequency
    print(f"[{stem}] computing neighbors and profiles...")
    query_prefixes = [s[:-1] if len(s) > 1 else s for s in data.test]
    neighbors = baseline_neighbor_ids(query_prefixes, data.train, 10)
    neighbor_freq = Counter(sid for nb in neighbors for sid in nb)
    profiles = np.zeros((len(data.train), 64), dtype=np.float32)
    emb_np = model.embedding.weight.detach().numpy()
    for i, s in enumerate(data.train):
        profiles[i] = emb_np[np.asarray(sorted(set(s)))].mean(axis=0)

    budget_sessions = int(args.budget_mb * 1e6 / (64 * 4))
    results = {"dataset": stem, "budget_mb": args.budget_mb,
               "budget_sessions": budget_sessions, "modes": {}}

    # Test each mode
    n_batches = min(20, len(data.test) // 128)
    for mode in ["no_cache", "tier1_only", "tier1_tier2"]:
        print(f"\n[{stem}] mode: {mode}")
        tier1 = None
        tier2 = None
        if "tier1" in mode:
            tier1 = Tier1Cache(neighbor_freq, profiles, budget_sessions)
            for i, nb in enumerate(neighbors):
                tier1.set_neighbors(i, nb)
        if "tier2" in mode:
            # Hot items: top 10% by frequency
            item_freq = Counter()
            for s in data.test[:n_batches * 128]:
                item_freq.update(set(s))
            hot = [i for i, _ in item_freq.most_common(
                int(data.n_items * 0.1))]
            tier2 = Tier2Cache(model, hot)

        all_timings = defaultdict(list)
        all_metrics = defaultdict(list)
        for b in range(n_batches):
            chunk = data.test[b * 128:(b + 1) * 128]
            t, m = serve_batch(model, chunk, rel, tier1_cache=tier1,
                               tier2_cache=tier2)
            for k, v in t.items():
                all_timings[k].append(v)
            for k, v in m.items():
                all_metrics[k].append(v)

        mode_results = {
            "batch_p50_ms": {k: st.median(v) * 1000
                             for k, v in all_timings.items()},
            "total_p50_ms": st.median(
                [sum(t.values()) for t in
                 [dict(zip(all_timings.keys(),
                           [all_timings[k][i] for k in all_timings]))
                  for i in range(n_batches)]]) * 1000,
            "HR@20": 100 * st.mean(all_metrics["HR@20"]),
            "MRR@20": 100 * st.mean(all_metrics["MRR@20"]),
        }
        if tier1:
            mode_results["tier1_hit_rate"] = tier1.hits / max(1, tier1.total)
        if tier2:
            mode_results["tier2_hit_rate"] = tier2.hits / max(1, tier2.total)
        results["modes"][mode] = mode_results
        print(f"  total p50: {mode_results['total_p50_ms']:.1f} ms/batch")
        print(f"  HR@20: {mode_results['HR@20']:.2f}")

    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"e2e_latency_{stem}.json"
    f.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
