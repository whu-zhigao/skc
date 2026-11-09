#!/usr/bin/env python3
"""B5: tier budget allocation under a fixed total memory budget. For
each split of the total across Tier-1 profiles / Tier-2 per-item
projections / Tier-3 affinity scores, the cache is built under the
chosen admission policy (nfa greedy on stream neighbor frequency /
cars / uniform, or one of the ONLINE admission families -- lru, arc,
tinylfu, wtinylfu, lecar, greedydualsize -- replayed over the stream
and materialized as a resident set) and the query stream is replayed
in ARRIVAL ORDER (B1 machinery). Reported: neighbor hit@k, Recall@20
vs exact full-corpus top-20, and P99 replay latency. The endogenous
NFA-greedy allocation (the paper's allocation: marginal-gain greedy
over tier bytes, admission fixed to NFA) runs as a labeled baseline;
--byte-matched re-runs that allocation with Tier-1 rows at reduced
precision so hit-rate/P99 compare at MATCHED bytes across bit widths.

Byte model (analytic, recorded in the JSON): Tier-1 profile = d*4 B
per session; Tier-2 = 3*d*4 B per covered item (Q/K/V projections);
Tier-3 = 4 B per affinity-score slot (one slot covers one item row).
Visibility model: a candidate session is scored only over the items
whose Tier-2 projections are resident; a query whose Tier-3 resident
item rows cover < 50% of its items falls back to scoring over the
resident subset only. All scoring is exact Jaccard on visible sets
(numpy; no embeddings needed, mirroring B1)."""
import argparse
import heapq
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ADMISSION_CITATIONS, _row_bytes, cars_admission_score,
                    config_out, jaccard_topk, neighbor_ids, serving_prefixes,
                    simulate_admitted, write_json)

T1_BYTES = None  # filled in main() from --d
T2_BYTES = None
T3_BYTES = 4.0
# online admission families (stream-replay admission sets; citations in
# common.ADMISSION_CITATIONS)
ONLINE_CHOICES = ("lru", "arc", "tinylfu", "wtinylfu", "lecar",
                  "greedydualsize")


def parse_split(spec: str, total_mb: float):
    if spec == "uniform":
        f = (1 / 3, 1 / 3, 1 / 3)
    else:
        parts = [float(x) for x in spec.split(",")]
        assert len(parts) == 3 and abs(sum(parts) - 1) < 1e-6, \
            f"bad split spec {spec!r} (need f1,f2,f3 summing to 1)"
        f = tuple(parts)
    return tuple(round(fi * total_mb, 4) for fi in f)


def admit(policy, n, cap1, freq_arr, gain_arr, cost_arr, rng,
          nbrs=None, seed=0):
    if policy == "nfa":
        score = freq_arr
    elif policy == "cars":
        score = cars_admission_score(freq_arr, gain_arr, cost_arr, 0.0)
    elif policy in ONLINE_CHOICES:
        # online admission policies: replay the query-stream neighbor
        # ids in arrival order through the policy cache and materialize
        # its final resident set (memoized per capacity).
        key = (policy, int(cap1))
        cache = admit._online.setdefault(key, simulate_admitted(
            policy, int(cap1), nbrs, seed=seed))
        return cache
    else:  # uniform
        sel = rng.choice(n, size=min(cap1, n), replace=False)
        return np.sort(sel)
    order = np.argsort(-score, kind="stable")
    return np.sort(order[:min(cap1, n)])


admit._online = {}   # (policy, cap1) -> resident set (arrival-order sim)


def build_cache(admitted, sessions, cap2, item_rows):
    """Walk admitted sessions in sid order (deterministic), covering
    per-item Tier-2 slots in one pass; returns (vis_len, prefix_items,
    materialized). Tier-3 residency = top item rows by query-stream
    frequency."""
    covered, running, mat = {}, 0, set()
    for sid in admitted:
        its = sorted(set(sessions[sid]))
        take = min(len(its), max(0, int(cap2) - running))
        covered[sid] = its[:take]
        running += take
        if take == len(its) and len(its) > 0:
            mat.add(int(sid))
    return covered, mat, item_rows


def replay(nbrs, queries, gt20, admitted, covered, mat, item_rows,
           k=10, k20=20, time_queries=True):
    admitted_set = set(int(s) for s in admitted)
    post = defaultdict(list)
    for sid in admitted:
        for it in covered[sid]:
            post[it].append(int(sid))
    vis_len = {int(s): len(covered[s]) for s in admitted}
    t1_hits = mat_hits = total_nb = 0
    recalls, elig, lat = [], [], []
    for i, (nb, q) in enumerate(zip(nbrs, queries)):
        for sid in nb:
            total_nb += 1
            t1_hits += int(int(sid) in admitted_set)
            mat_hits += int(int(sid) in mat)
        q_items = set(q)
        cov = len(q_items & item_rows)
        e = cov >= 0.5 * max(1, len(q_items))
        elig.append(float(e))
        q_vis = q_items if e else (q_items & item_rows)
        t0 = time.perf_counter()
        cnt = Counter()
        for it in q_vis:
            for sid in post.get(it, ()):
                cnt[sid] += 1
        nq = len(q_vis)
        scored = ((sid, c / (nq + vis_len[sid] - c)) for sid, c in
                  cnt.items()) if nq else ()
        top = {sid for sid, _ in heapq.nlargest(
            k20, scored, key=lambda x: (x[1], -x[0]))}
        lat.append(time.perf_counter() - t0)
        recalls.append(len(top & set(gt20[i])) / k20)
    out = {"hit_at_k_tier1": t1_hits / max(1, total_nb),
           "hit_at_k_materialized": mat_hits / max(1, total_nb),
           "recall_at_20": float(np.mean(recalls)),
           "eligible_frac": float(np.mean(elig)),
           "latency_ms": {"p50": 1000 * float(np.percentile(lat, 50)),
                          "p99": 1000 * float(np.percentile(lat, 99)),
                          "mean": 1000 * float(np.mean(lat))}}
    return out


def main():
    global T1_BYTES, T2_BYTES
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "--data", dest="dataset", type=Path,
                    required=True)
    ap.add_argument("--total-mb", type=float, default=29.0)
    ap.add_argument("--admission",
                    choices=("nfa", "cars", "uniform") + ONLINE_CHOICES,
                    default="nfa",
                    help="nfa/cars/uniform = offline one-shot admission "
                         "on full-trace stats; the online family "
                         "(lru,arc,tinylfu,wtinylfu,lecar,greedydualsize"
                         ") replays the query-stream neighbor ids in "
                         "arrival order and materializes the policy's "
                         "final resident set (citations in "
                         "common.ADMISSION_CITATIONS)")
    ap.add_argument("--byte-matched", action="store_true",
                    help="byte-matched precision comparison: re-run the "
                         "endogenous greedy allocation at the same "
                         "total budget with Tier-1 rows at reduced "
                         "precision (--match-bits), so hit-rate/P99 "
                         "compare at MATCHED bytes across bit widths "
                         "(tab:split hit-rate+P99 columns); one JSON "
                         "per bit width (skips the fixed-split sweep)")
    ap.add_argument("--match-bits", default="8,4,2",
                    help="comma list of Tier-1 bit widths for "
                         "--byte-matched (default 8,4,2)")
    ap.add_argument("--tier1-bytes", type=float, default=None,
                    help="override the analytic Tier-1 bytes/session "
                         "(for mixed-precision byte-matched runs)")
    ap.add_argument("--precision-label", default=None,
                    help="label for a custom --tier1-bytes run "
                         "(e.g., utility-mixed)")
    ap.add_argument("--splits", action="append", default=None,
                    metavar="SPEC",
                    help="split spec: 'uniform' or 'f1,f2,f3' (sum 1); "
                         "repeatable; default suite = uniform + 3 curves")
    ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--greedy-subsample", type=int, default=200)
    ap.add_argument("--greedy-step-mb", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun configs whose per-config JSON exists")
    args = ap.parse_args()

    if not args.dataset.exists():
        sys.exit(f"[b5] dataset not found: {args.dataset}\n"
                 f"       pass --dataset <repo>/code/data/diginetica.pkl "
                 f"(or --data, same flag)")
    from common import load_data
    rng = np.random.default_rng(args.seed)
    stem = args.dataset.stem

    def split_pct(spec):
        f = ((1 / 3, 1 / 3, 1 / 3) if spec == "uniform"
             else tuple(float(x) for x in spec.split(",")))
        return "-".join(str(int(round(x * 100))) for x in f)

    specs = args.splits or ["uniform", "0.6,0.3,0.1", "0.5,0.3,0.2",
                            "0.2,0.4,0.4"]
    tag_base = (f"{stem}_{{spec}}_{args.admission}_t{args.total_mb:g}"
                f"_nq{args.n_query}_s{args.seed}")
    tag_greedy = (f"{stem}_{args.admission}-greedy_t{args.total_mb:g}"
                  f"_nq{args.n_query}_s{args.seed}")
    match_bits = [int(x) for x in args.match_bits.split(",")
                  if x.strip()] if args.byte_matched else []
    custom_precision = (args.tier1_bytes is not None
                        and args.precision_label is not None)
    if custom_precision:
        match_bits = [8]  # placeholder; bytes are set explicitly below
        bm_tags = [f"{stem}_bm-{args.precision_label}_{args.admission}"
                   f"_t{args.total_mb:g}_nq{args.n_query}_s{args.seed}"]
    else:
        bm_tags = ([f"{stem}_bm-int{b}_{args.admission}"
                   f"_t{args.total_mb:g}_nq{args.n_query}_s{args.seed}"
                   for b in match_bits] if args.byte_matched else [])
    pending, paths = [], {}
    if args.byte_matched:
        run_greedy = args.force or args.out
        if not args.out:
            for b, tg in zip(match_bits, bm_tags):
                p, skip = config_out("b5", tg, None, args.force)
                if not skip:
                    run_greedy = True
            if not run_greedy:
                print("[b5] all byte-matched configs exist, nothing to "
                      "do (--force to rerun)")
                return
    else:
        for spec in specs:
            tag = tag_base.format(spec=split_pct(spec))
            p, skip = config_out("b5", tag, None, args.force)
            paths[spec] = p
            if not skip:
                pending.append(spec)
        p_greedy, skip_g = config_out("b5", tag_greedy, None, args.force)
        run_greedy = args.force or args.out or not skip_g
        if args.out:
            pending = specs          # legacy mode: run everything
        if not pending and not run_greedy:
            print("[b5] all configs exist, nothing to do "
                  "(--force to rerun)")
            return
    T1_BYTES = args.d * 4.0
    T2_BYTES = 3 * args.d * 4.0
    data = load_data(args.dataset)
    n = len(data.train)
    nq = min(args.n_query, len(data.test))
    queries = serving_prefixes(data.test[:nq])
    print(f"[{stem}] {n} sessions, {nq} queries; "
          f"admission={args.admission}, total={args.total_mb}MB")

    nbrs = neighbor_ids(queries, data.train, args.k)
    gt20 = jaccard_topk(queries, data.train, 20)
    freq = Counter(sid for nb in nbrs for sid in nb)
    freq_arr = np.zeros(n)
    for sid, c in freq.items():
        freq_arr[sid] = c
    gain_arr = np.fromiter((len(set(s)) for s in data.train),
                           dtype=np.float64, count=n)
    cost_arr = np.fromiter((len(s) for s in data.train),
                           dtype=np.float64, count=n)
    q_item_freq = Counter(it for q in queries for it in set(q))

    def run_split(mb, qs=nbrs, gts=gt20, time_q=True):
        mb1, mb2, mb3 = mb
        cap1 = int(mb1 * 1e6 / T1_BYTES)
        cap2 = int(mb2 * 1e6 / T2_BYTES)
        cap3 = int(mb3 * 1e6 / T3_BYTES)
        admitted = admit(args.admission, n, cap1, freq_arr, gain_arr,
                         cost_arr, rng, nbrs=nbrs, seed=args.seed)
        rows = [it for it, _ in q_item_freq.most_common(cap3)]
        covered, mat, item_rows = build_cache(admitted, data.train, cap2,
                                              set(rows))
        r = replay(qs, queries, gts, admitted, covered, mat, item_rows,
                   k=args.k, time_queries=time_q)
        r.update({"mb": [mb1, mb2, mb3],
                  "cap": [len(admitted), cap2, cap3],
                  "materialized": len(mat)})
        return r

    def header(extra=None):
        h = {"dataset": stem, "n_sessions": n, "n_query": nq,
             "k": args.k, "d": args.d, "arrival_order": True,
             "total_mb": args.total_mb, "admission": args.admission,
             "admission_citation": ADMISSION_CITATIONS.get(
                 args.admission, args.admission),
             "seed": args.seed,
             "byte_model": {"tier1_per_session": T1_BYTES,
                            "tier2_per_item": T2_BYTES,
                            "tier3_per_score": T3_BYTES},
             "note": "NFA = greedy top-cap1 on full-trace neighbor "
                     "frequency (B1 oracle); CARS uses gain=len(unique "
                     "items), cost=len(session) (Tier-2 footprint proxy),"
                     " staleness=0 (see B7 for drift); uniform = seeded "
                     "random. Scoring is exact Jaccard over "
                     "Tier-2-visible sets; queries with <50% Tier-3 "
                     "item-row coverage fall back to the resident "
                     "subset. Latency = per-query CPU replay scoring on "
                     "this node."}
        if extra:
            h.update(extra)
        return h

    # endogenous NFA-greedy allocation (paper allocation): marginal-gain
    # greedy over tier bytes, admission fixed to NFA, objective = hit@k
    # on materialized neighbors + 0.25 * full-scoring eligibility,
    # evaluated on the greedy subsample.
    def run_greedy_alloc(write_path, extra=None):
        sub = slice(0, min(args.greedy_subsample, nq))

        def eval_alloc(alloc):
            r = run_split(alloc, qs=nbrs[sub], gts=gt20[sub], time_q=False)
            return (r["hit_at_k_materialized"]
                    + 0.25 * r["eligible_frac"], r)

        step, floor = args.greedy_step_mb, 0.5
        alloc = [floor, floor, floor]
        remaining = args.total_mb - sum(alloc)
        n_eval = 0
        while remaining >= step - 1e-9:
            best = None
            for t in range(3):
                cand = alloc.copy()
                cand[t] += step
                v, _ = eval_alloc(cand)
                n_eval += 1
                if best is None or v > best[0]:
                    best = (v, t)
            alloc[best[1]] += step
            remaining -= step
        v, _ = eval_alloc(alloc)
        r = run_split(tuple(round(a, 4) for a in alloc))  # full stream
        r["split"] = "nfa-greedy(endogenous)"
        r["greedy_alloc_mb"] = [round(a, 4) for a in alloc]
        r["greedy_objective"] = float(v)
        r["greedy_evals"] = n_eval
        greedy_meta = {"step_mb": step, "floor_mb": floor,
                       "objective": "hit_at_k_materialized + "
                                    "0.25*eligible_frac (subsampled)",
                       "evals": n_eval}
        print(f"  {'nfa-greedy':>14s}: {alloc} hitM "
              f"{r['hit_at_k_materialized']:.3f} "
              f"R@20 {r['recall_at_20']:.3f} "
              f"p99 {r['latency_ms']['p99']:.2f}ms")
        payload = header({"config": "nfa-greedy(endogenous)",
                          "greedy": greedy_meta, "result": r})
        if extra:
            payload.update(extra)
        if write_path is not None:
            write_json(write_path, payload)
        return r, greedy_meta

    rows = []
    if args.byte_matched:
        # byte-matched precision comparison (--byte-matched): Tier-1
        # rows quantized to `bits` (analytic byte model from
        # common._row_bytes), same total budget, endogenous greedy
        # allocation per width -- hit-rate/P99 compare at MATCHED bytes.
        for b, tg in zip(match_bits, bm_tags):
            T1_BYTES = (args.tier1_bytes if custom_precision
                        else _row_bytes(args.d, b))
            if args.out:
                p_bm = Path(str(args.out) + (
                    f".int{b}" if len(match_bits) > 1 else ""))
            else:
                p_bm = config_out("b5", tg)[0]
                if p_bm.exists() and not args.force:
                    print(f"[b5] bm-int{b}: exists, skipping {p_bm}")
                    continue
            print(f"[b5] byte-matched tier1 bits={b} "
                  f"({T1_BYTES} B/session)")
            run_greedy_alloc(p_bm, extra={
                "config": "byte-matched",
                "tier1_bits": (None if custom_precision else b),
                "tier1_bytes": (args.tier1_bytes if custom_precision
                                else _row_bytes(args.d, b)),
                "precision_label": (args.precision_label
                                    if custom_precision else None),
                "note_byte_matched": "Tier-1 per-session bytes set to "
                                     f"{T1_BYTES} "
                                     + (f"({args.precision_label}); "
                                        if custom_precision else
                                        f"(int{b} codes + fp32 scale); ")
                                     + "same total budget and allocation "
                                       "machinery as the fp32 run -- "
                                       "hit-rate/P99 compare at matched "
                                       "bytes across widths"})
        return

    for spec in pending:
        mb = parse_split(spec, args.total_mb)
        r = run_split(mb)
        r["split"] = spec
        rows.append(r)
        print(f"  {spec:>14s}: hit1@{args.k} {r['hit_at_k_tier1']:.3f} "
              f"hitM {r['hit_at_k_materialized']:.3f} "
              f"R@20 {r['recall_at_20']:.3f} "
              f"p99 {r['latency_ms']['p99']:.2f}ms")
        if not args.out:
            write_json(paths[spec], header(
                {"config": "fixed-split", "split": spec, "result": r}))

    g_meta = None
    if run_greedy:
        g_row, g_meta = run_greedy_alloc(None if args.out else p_greedy)
        rows.append(g_row)

    md = ["| split | admission | mb(T1/T2/T3) | hit1@k | hitMat@k | "
          "R@20 | p99(ms) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['split']} | {args.admission} | "
            f"{r['mb'][0]:.1f}/{r['mb'][1]:.1f}/{r['mb'][2]:.1f} | "
            f"{r['hit_at_k_tier1']:.3f} | "
            f"{r['hit_at_k_materialized']:.3f} | "
            f"{r['recall_at_20']:.3f} | "
            f"{r['latency_ms']['p99']:.2f} |")
    print("\n".join(md))

    if args.out:   # legacy single-file mode: combined JSON
        write_json(args.out, header(
            {"greedy": g_meta, "splits": rows, "markdown_rows": md}))


if __name__ == "__main__":
    main()
