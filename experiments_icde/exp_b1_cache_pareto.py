#!/usr/bin/env python3
"""B1: Tier-1 hit-rate vs memory-budget Pareto across admission
strategies, replayed in ARRIVAL ORDER. Strategies: oracle/nfa-offline
(greedy top-B on full-trace neighbor frequency), nfa-sketch (online
Count-Min), and the online admission family -- lru, lfu (lookup
frequency, periodic re-admission), arc, tinylfu, wtinylfu, lecar,
greedydualsize (citations in common.ADMISSION_CITATIONS, also recorded
in the JSON) -- plus the paged-nfa page-size sweep and the exact
full-cache reference point (hit = 1.0 at the budget that materializes
every unique neighbor). Metadata overheads are computed analytically
and reported in the JSON."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ADMISSION_CITATIONS, CountMin,
                    config_out, make_admission_cache, serving_prefixes,
                    write_json)

ALL_POLICIES = ("lru", "lfu", "arc", "tinylfu", "wtinylfu", "lecar",
                "greedydualsize")


def stream_hits_online(nbrs, capacity, policy, window=256, seed=0):
    if policy == "lfu":
        cache, counts = set(), Counter()

        def req(x):
            counts[x] += 1
            return x in cache
    else:
        req = make_admission_cache(policy, capacity, seed=seed).request
    hits = total = 0
    for i, nb in enumerate(nbrs):
        if policy == "lfu" and i % window == 0 and counts:
            cache = {sid for sid, _ in counts.most_common(capacity)}
        for sid in nb:
            total += 1
            hits += int(req(sid))
    return hits / max(1, total)


def stream_nfa_sketch(nbrs, capacity, w, d):
    cm, cached = CountMin(w=w, d=d), set()
    hits = total = 0
    seen = np.empty(0, dtype=np.int64)
    for i, nb in enumerate(nbrs):
        cm.update(nb)
        seen = np.unique(np.concatenate(
            (seen, np.asarray(nb, dtype=np.int64))))
        for sid in nb:
            total += 1
            hits += int(sid in cached)
        # A serving node knows the retrieved neighbor set after answering the
        # request.  Prefetch those rows immediately, subject to capacity; the
        # periodic sketch refresh later chooses eviction victims.
        if len(cached) < capacity:
            cached.update(int(sid) for sid in nb
                          if len(cached) < capacity)
        if i % 256 == 255 or i == len(nbrs) - 1:
            est = cm.estimate(seen)
            cached = set(seen[np.argsort(-est)[:capacity]].tolist())
    return hits / max(1, total), cm.bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-query", type=int, default=3000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--admission", default="all",
                    metavar="POLICIES",
                    help="'all' (default: lru,lfu,arc,tinylfu,wtinylfu,"
                         "lecar,greedydualsize) or a comma list; "
                         "citations in common.ADMISSION_CITATIONS")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the stochastic admission policies "
                         "(lecar draws, Count-Min sketches)")
    ap.add_argument("--sketch-w", type=int, default=4096)
    ap.add_argument("--sketch-d", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()

    if args.admission.strip().lower() == "all":
        policies = list(ALL_POLICIES)
    else:
        policies = [p.strip().lower() for p in args.admission.split(",")
                    if p.strip()]
    bad = [p for p in policies if p not in ALL_POLICIES]
    if bad:
        sys.exit(f"[b1] unknown admission policies: {bad} "
                 f"(valid: {ALL_POLICIES})")
    stem = args.data.stem
    # default run keeps the historical tag (backward compatible with
    # existing files); a custom policy set or seed is encoded in the tag.
    tag = f"{stem}_nq{args.n_query}_k{args.k}_d{args.d}"
    if policies != list(ALL_POLICIES):
        tag += "_adm-" + "+".join(policies)
    if args.seed != 0:
        tag += f"_s{args.seed}"
    out_path, skip = config_out("b1", tag, args.out, args.force)
    if skip:
        print(f"[b1] exists, skipping: {out_path} (--force to rerun)")
        return

    from common import load_data, neighbor_ids
    data = load_data(args.data)
    nbrs = neighbor_ids(serving_prefixes(
                            data.test[:min(args.n_query, len(data.test))]),
                        data.train, args.k)
    freq = Counter(sid for nb in nbrs for sid in nb)
    total_look = max(1, sum(len(nb) for nb in nbrs))
    budgets_mb = [1, 2, 5, 10, 20, 50, 100]
    page_sizes = [16, 32, 64, 128, 256]

    rows = []
    for mb in budgets_mb:
        cap = int(mb * 1e6 / (args.d * 4))
        row = {"budget_mb": mb, "budget_sessions": cap}
        top = {sid for sid, _ in freq.most_common(cap)}
        row["oracle"] = row["nfa_offline"] = sum(
            1 for nb in nbrs for s in nb if s in top) / total_look
        for pol in policies:
            row[pol] = stream_hits_online(nbrs, cap, pol, seed=args.seed)
        row["nfa_sketch"], row["sketch_bytes"] = stream_nfa_sketch(
            nbrs, cap, args.sketch_w, args.sketch_d)
        row["meta_bytes_session_level"] = cap * 24
        rows.append(row)
        print(f"  {mb:3d}MB cap={cap:6d}: " + " ".join(
            f"{kk}={vv:.3f}" for kk, vv in row.items()
            if isinstance(vv, float)))

    paged = []
    for G in page_sizes:
        for mb in budgets_mb:
            cap = int(mb * 1e6 / (args.d * 4))
            pf = Counter()
            for sid, c in freq.items():
                pf[sid // G] += c
            pages = {p for p, _ in pf.most_common(max(1, cap // G))}
            hit = sum(1 for nb in nbrs for s in nb if s // G in pages) \
                / total_look
            paged.append({"G": G, "budget_mb": mb, "hit": hit,
                          "pages": len(pages),
                          "meta_bytes": len(pages) * 16 + cap * args.d * 4})

    # exact reference point (upper bound): materialize EVERY unique
    # neighbor at fp32 -- every lookup in the replay stream then hits.
    exact_ref = {
        "budget_sessions": len(freq),
        "budget_mb": round(len(freq) * args.d * 4 / 1e6, 3),
        "hit": 1.0,
        "note": "exact full-cache reference: neighbor ids come from the "
                "exact-Jaccard top-k reference path (common.neighbor_ids); "
                "materializing all active neighbors makes every lookup "
                "hit by construction (Pareto upper bound)"}

    write_json(out_path, {
        "dataset": stem, "n_query": len(nbrs), "k": args.k,
        "arrival_order": True, "budgets": rows, "paged": paged,
        "active_unique": len(freq),
        "admissions": {p: ADMISSION_CITATIONS.get(p, p) for p in policies},
        "seed": args.seed,
        "exact_reference": exact_ref,
        "note": "oracle==nfa_offline by construction (greedy top-B on "
                "full-trace neighbor frequency); the online gap is "
                "carried by nfa_sketch vs the online admission family. "
                "Admission citations: " + "; ".join(
                    f"{p} = {ADMISSION_CITATIONS.get(p, p)}"
                    for p in policies if p in ADMISSION_CITATIONS)})


if __name__ == "__main__":
    main()
