#!/usr/bin/env python3
"""A1: neighbor-graph reuse characterization. Measures active-neighbor
fraction, frequency distribution (Gini, top-m coverage), and hub-set
stability across temporal windows of the query stream (tests the
persistence-of-importance assumption behind NFA/sketch admission)."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config_out, gini, neighbor_ids, serving_prefixes, write_json


def topm_overlap(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-query", type=int, default=3000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-windows", type=int, default=10)
    ap.add_argument("--topm", type=int, default=2000)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()

    stem = args.data.stem
    out_path, skip = config_out(
        "a1", f"{stem}_nq{args.n_query}_k{args.k}_w{args.n_windows}",
        args.out, args.force)
    if skip:
        print(f"[a1] exists, skipping: {out_path} (--force to rerun)")
        return

    from common import load_data
    data = load_data(args.data)
    nq = min(args.n_query, len(data.test))
    print(f"[{stem}] neighbors for {nq} queries...")
    nbrs = neighbor_ids(serving_prefixes(data.test[:nq]), data.train, args.k)

    freq = Counter(sid for nb in nbrs for sid in nb)
    total = sum(len(nb) for nb in nbrs)
    m = np.array(sorted(freq.values(), reverse=True))
    coverage = {frac: float(m[:max(1, int(len(m) * frac))].sum() / m.sum())
                for frac in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)}

    W = args.n_windows
    edges = np.linspace(0, nq, W + 1, dtype=int)
    hub_sets = []
    for w in range(W):
        wf = Counter(sid for nb in nbrs[edges[w]:edges[w + 1]] for sid in nb)
        hub_sets.append({sid for sid, _ in wf.most_common(args.topm)})
    stab = [topm_overlap(hub_sets[i], hub_sets[i + 1])
            for i in range(W - 1)]
    global_hubs = {sid for sid, _ in freq.most_common(args.topm)}
    glob_overlap = [topm_overlap(global_hubs, h) for h in hub_sets]

    write_json(out_path, {
        "dataset": stem, "n_query": nq, "k": args.k,
        "corpus_size": len(data.train),
        "active_fraction": len(freq) / len(data.train),
        "unique_neighbor_sessions": len(freq),
        "total_lookups": total, "gini": gini(freq.values()),
        "top_coverage": coverage,
        "hub_stability": {
            "windows": W, "topm": args.topm,
            "consecutive_jaccard": stab,
            "mean_consecutive_jaccard": float(np.mean(stab)),
            "global_vs_window_jaccard": glob_overlap,
            "mean_global_vs_window": float(np.mean(glob_overlap))}})


if __name__ == "__main__":
    main()
