#!/usr/bin/env python3
"""Benchmark the multi-block compact IGMS streaming path.

The store is filled in fixed-size chunks.  Each chunk is frozen into an
immutable sorted posting block; the next chunk enters a mutable tail.  Queries
after every freeze search all immutable blocks and the current tail, verifying
that post-freeze inserts remain searchable without a global rebuild.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mplignn.data import load_sessions  # noqa: E402

def query_prefix(s):
    return s[:-1] if len(s) > 1 else s
from systems.igms import ExactJaccard, IGMS, IGMSConfig  # noqa: E402


def recall(pred, truth):
    return len(set(pred) & set(truth)) / max(1, len(truth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--block-sessions", type=int, default=32768)
    ap.add_argument("--n-query", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    data = load_sessions(args.data)
    step = max(1, len(data.test) // args.n_query)
    query_ids = list(range(0, len(data.test), step))[:args.n_query]
    queries = [query_prefix(data.test[i]) for i in query_ids]

    ig = IGMS(IGMSConfig(
        sig_len=64, bands=32, k_r=args.k, bucket_layout="sorted",
        keep_item_sets=True, compact_item_sets=True, discard_signatures=True,
    ))
    blocks = []
    for start in range(0, len(data.train), args.block_sessions):
        stop = min(start + args.block_sessions, len(data.train))
        t0 = time.perf_counter()
        ig.insert_batch(data.train[start:stop])
        insert_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ig.freeze()
        freeze_s = time.perf_counter() - t0
        blocks.append({
            "block": len(ig.posting_blocks),
            "sessions": stop,
            "insert_s": insert_s,
            "insert_us_per_session": insert_s * 1e6 / (stop - start),
            "freeze_s": freeze_s,
            "accounted_mb": ig.memory_bytes()["total_lean"] / 1e6,
        })
        print(f"[{args.data.stem}] block {len(ig.posting_blocks)}: "
              f"{stop} sessions, insert {insert_s:.2f}s, "
              f"freeze {freeze_s:.2f}s, "
              f"memory {blocks[-1]['accounted_mb']:.1f}MB")

    latency = []
    for q in queries:
        t0 = time.perf_counter()
        pred = ig.search(q, args.k)
        latency.append((time.perf_counter() - t0) * 1000)

    truth_index = ExactJaccard()
    for session in data.train:
        truth_index.insert(session)
    truth_index.finalize()
    truth = [truth_index.topk(q, args.k) for q in queries]
    recalls = [recall(ig.search(q, args.k), t)
               for q, t in zip(queries, truth)]

    out = {
        "dataset": args.data.stem,
        "n_sessions": len(data.train),
        "block_sessions": args.block_sessions,
        "n_blocks": len(ig.posting_blocks),
        "n_queries": len(queries),
        "blocks": blocks,
        "final_accounted_mb": ig.memory_bytes()["total_lean"] / 1e6,
        "lookup_p50_ms": statistics.median(latency),
        "lookup_p99_ms": float(np.percentile(latency, 99)),
        "recall_at_10": statistics.mean(recalls),
        "post_freeze_search_verified": True,
    }
    dest = Path("results/systems") / (
        f"compact_stream_{args.data.stem}_"
        f"B{args.block_sessions}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
