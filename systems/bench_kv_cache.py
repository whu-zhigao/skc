#!/usr/bin/env python3
"""Benchmark E20: Session KV Cache hit-rate optimization.

Measures three strategies against LRU and oracle:
  NFA  (Neighbor-Frequency Admission) — H2O-style: admit sessions that
       appear most often as neighbors, not that are queried most often
  PPS  (Paged Profile Storage)        — PagedAttention-style: page-level
       admission/eviction, exploiting intra-page sharing
  DAI  (Delta-Aware Invalidation)     — during training, only invalidate
       pages whose items' embeddings drifted beyond θ

Output: hit-rate vs memory-budget Pareto curves per strategy.

Run: python3 systems/bench_kv_cache.py --data data/diginetica.pkl
Writes results/systems/kv_cache_<stem>.json
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from mplignn.data import load_sessions  # noqa: E402
from mplignn.model import MPLI_GNN  # noqa: E402
from systems.bench_retrieval import baseline_neighbor_ids  # noqa: E402
from systems.igms import IGMS, IGMSConfig  # noqa: E402
from systems.skc import OracleCache, SessionKVCache, SKCConfig  # noqa: E402

REAL = Path(__file__).parent.parent / "results" / "real"


def run_cache_strategies(neighbors, profiles, budgets_mb):
    """Sweep memory budgets; for each, run NFA / LRU / Paged-NFA / Oracle."""
    n_queries = len(neighbors)
    all_neighbor_ids = [sid for nb in neighbors for sid in nb]
    neighbor_freq = Counter(all_neighbor_ids)
    unique_sessions = sorted(neighbor_freq.keys())
    d = profiles.shape[1]

    results = {"budgets": [], "strategies": {}}

    for budget_mb in budgets_mb:
        entry = {"budget_mb": budget_mb}
        nbytes_per_session = d * 4  # fp32
        budget_sessions = int(budget_mb * 1e6 / nbytes_per_session)
        entry["budget_sessions"] = budget_sessions

        # --- Oracle (offline optimal: top-K by neighbor frequency)
        oracle = OracleCache(budget_sessions)
        oracle.plan(neighbor_freq)
        entry["oracle_hit_rate"] = oracle.hit_rate(neighbors)

        # --- NFA (session-level LFU on neighbor frequency)
        # Admit the top-K sessions by neighbor-appearance frequency.
        top_sessions = [sid for sid, _ in
                        neighbor_freq.most_common(budget_sessions)]
        nfa_cached = set(top_sessions)
        nfa_hits = sum(1 for nb in neighbors for sid in nb
                       if sid in nfa_cached)
        entry["nfa_hit_rate"] = nfa_hits / max(1, sum(len(nb) for nb in neighbors))

        # --- LRU (session-level, recency-based admission)
        # Simulate LRU: process queries in order, admit recently accessed
        lru_cache = {}
        lru_order = []
        for nb in neighbors:
            for sid in nb:
                if sid in lru_cache:
                    lru_order.remove(sid)
                    lru_order.append(sid)
                else:
                    if len(lru_cache) >= budget_sessions:
                        evict = lru_order.pop(0)
                        del lru_cache[evict]
                    lru_cache[sid] = True
                    lru_order.append(sid)
        lru_hits = sum(1 for nb in neighbors for sid in nb
                       if sid in lru_cache)
        entry["lru_hit_rate"] = lru_hits / max(1, sum(len(nb) for nb in neighbors))
        entry["lru_final_size"] = len(lru_cache)

        # --- Paged-NFA (page-level admission + NFA scoring)
        # Pages of 128 sessions; admit pages whose aggregate neighbor
        # frequency is highest (maximizes per-page hit density)
        page_scores = defaultdict(float)
        for sid, freq in neighbor_freq.items():
            page_scores[sid // 128] += freq
        top_pages = sorted(page_scores, key=page_scores.get, reverse=True)
        budget_pages = max(1, budget_sessions // 128)
        cached_pages = set(top_pages[:budget_pages])
        page_hits, page_total = 0, 0
        for nb in neighbors:
            for sid in nb:
                page_total += 1
                if sid // 128 in cached_pages:
                    page_hits += 1
        entry["paged_nfa_hit_rate"] = page_hits / max(1, page_total)
        entry["paged_nfa_pages"] = len(cached_pages)

        results["budgets"].append(entry)
        print(f"  budget {budget_mb:5.0f}MB ({budget_sessions:5d} sess): "
              f"oracle {entry['oracle_hit_rate']:.3f}  "
              f"NFA {entry['nfa_hit_rate']:.3f}  "
              f"LRU {entry['lru_hit_rate']:.3f}  "
              f"paged-NFA {entry['paged_nfa_hit_rate']:.3f}")

    return results


def run_dai_experiment(data, model, neighbors, profiles):
    """Delta-Aware Invalidation: simulate training steps and measure
    how many pages need invalidation under different thresholds."""
    emb_old = model.embedding.weight.detach().clone()
    # simulate one optimizer step's drift
    with torch.no_grad():
        emb_new = emb_old + torch.randn_like(emb_old) * 0.01

    # Map items to sessions containing them
    item_to_sessions = defaultdict(list)
    for sid, s in enumerate(data.train[:50000]):  # cap for speed
        for item in set(s):
            item_to_sessions[item].append(sid)

    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    dai_results = []
    for theta in thresholds:
        cache = SessionKVCache(SKCConfig(
            memory_budget_mb=100, d=64,
            invalidation_theta=theta, admission="nfa"))
        # populate cache
        cache.admit(list(range(0, 50000, 100)), profiles)
        n_before = len(cache.pages)
        invalidated = cache.invalidate_delta(emb_old, emb_new,
                                              item_to_sessions)
        dai_results.append({
            "theta": theta,
            "pages_before": n_before,
            "pages_invalidated": invalidated,
            "invalidation_rate": invalidated / max(1, n_before),
        })
        print(f"  DAI θ={theta:.3f}: {invalidated}/{n_before} pages "
              f"invalidated ({100*invalidated/max(1,n_before):.1f}%)")
    return dai_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=REAL / "abl_no_residual_diginetica.pt")
    args = ap.parse_args()
    data = load_sessions(args.data)
    stem = args.data.stem

    model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=2,
                     n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                     no_residual=True)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu",
                                     weights_only=True))
    model.eval()
    emb = model.embedding.weight.detach().numpy()

    # Pool profiles for the train corpus
    print(f"[{stem}] pooling profiles for {len(data.train)} sessions...")
    profiles = np.zeros((len(data.train), 64), dtype=np.float32)
    for i, s in enumerate(data.train):
        profiles[i] = emb[np.asarray(sorted(set(s)))].mean(axis=0)

    # Get neighbor sets for test queries
    n_query = min(3000, len(data.test))
    print(f"[{stem}] computing neighbor sets for {n_query} queries...")
    neighbors = baseline_neighbor_ids([s[:-1] if len(s) > 1 else s for s in data.test[:n_query]], data.train, 10)

    # Neighbor sharing statistics
    neighbor_freq = Counter(sid for nb in neighbors for sid in nb)
    unique = len(neighbor_freq)
    total_lookups = sum(len(nb) for nb in neighbors)
    print(f"[{stem}] neighbor sharing: {unique} unique / {total_lookups} "
          f"total lookups ({100*unique/total_lookups:.1f}% unique)")

    # S1+S2: hit-rate vs memory budget Pareto
    print(f"\n[{stem}] === Cache hit-rate vs memory budget ===")
    budgets = [5, 10, 20, 50, 100, 200, 500]
    cache_results = run_cache_strategies(neighbors, profiles, budgets)

    # S3: Delta-aware invalidation
    print(f"\n[{stem}] === Delta-Aware Invalidation ===")
    dai_results = run_dai_experiment(data, model, neighbors, profiles)

    out = {
        "dataset": stem,
        "n_query": n_query,
        "neighbor_sharing": {
            "unique_neighbor_sessions": unique,
            "total_lookups": total_lookups,
            "corpus_size": len(data.train),
            "active_fraction": unique / len(data.train),
        },
        "cache_strategies": cache_results,
        "delta_invalidation": dai_results,
    }
    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"kv_cache_{stem}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
