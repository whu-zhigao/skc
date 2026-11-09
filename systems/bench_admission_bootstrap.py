#!/usr/bin/env python3
"""Query-level bootstrap CIs for online cache-admission policies."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mplignn.data import load_sessions  # noqa: E402
from systems.bench_retrieval import baseline_neighbor_ids  # noqa: E402
from experiments_icde.common import (CountMin, make_admission_cache,  # noqa: E402
                                     serving_prefixes)


def policy_hits(nbrs, capacity, policy, seed=0):
    req = make_admission_cache(policy, capacity, seed=seed).request
    return np.asarray([
        sum(int(req(sid)) for sid in nb) / len(nb) for nb in nbrs
    ], dtype=np.float64)


def nfa_hits(nbrs, capacity, w=4096, d=4, refresh=256):
    cm, cached = CountMin(w=w, d=d), set()
    out = []
    seen = np.empty(0, dtype=np.int64)
    for i, nb in enumerate(nbrs):
        cm.update(nb)
        seen = np.unique(np.concatenate(
            (seen, np.asarray(nb, dtype=np.int64))))
        out.append(sum(sid in cached for sid in nb) / len(nb))
        if len(cached) < capacity:
            cached.update(int(sid) for sid in nb
                          if len(cached) < capacity)
        if i % refresh == refresh - 1 or i == len(nbrs) - 1:
            est = cm.estimate(seen)
            cached = set(seen[np.argsort(-est)[:capacity]].tolist())
    return np.asarray(out, dtype=np.float64)


def bootstrap_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-query", type=int, default=3000)
    ap.add_argument("--budget-mb", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    data = load_sessions(args.data)
    queries = serving_prefixes(
        data.test[:min(args.n_query, len(data.test))])
    nbrs = baseline_neighbor_ids(queries, data.train, args.k)
    capacity = int(args.budget_mb * 1e6 / (64 * 4))
    arms = {
        "sketch_nfa": nfa_hits(nbrs, capacity),
        "lru": policy_hits(nbrs, capacity, "lru"),
        "arc": policy_hits(nbrs, capacity, "arc"),
        "wtinylfu": policy_hits(nbrs, capacity, "wtinylfu"),
    }
    out = {"dataset": args.data.stem, "n_queries": len(nbrs),
           "budget_mb": args.budget_mb, "capacity": capacity, "arms": {}}
    for name, x in arms.items():
        out["arms"][name] = {
            "mean": float(x.mean()),
            "query_bootstrap_95_ci": bootstrap_ci(x),
        }
    dest = Path("results/systems") / (
        f"admission_bootstrap_{args.data.stem}_{args.budget_mb:g}mb.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
