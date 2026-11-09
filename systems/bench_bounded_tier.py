#!/usr/bin/env python3
"""Bounded-total-memory posting-store benchmark.

Builds a cold two-tier MinHash/CSR store, then serves leakage-free query
prefixes under explicit hot-memory caps. Cold band buckets and candidate item
records are read from disk with bounded temporary allocations; the benchmark
reports recall, latency, resident bytes, disk reads, and peak temporary bytes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_icde.common import serving_prefixes  # noqa: E402
from mplignn.data import load_sessions  # noqa: E402
from systems.bounded_tier import (BoundedTieredPostingStore,  # noqa: E402
                                  build_bounded_tier)
from systems.igms import ExactJaccard  # noqa: E402


def recall_at_k(pred, truth, k):
    return len(set(pred[:k]) & set(truth[:k])) / k


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n-sessions", type=int, default=50_000)
    parser.add_argument("--n-query", type=int, default=100)
    parser.add_argument("--block-sessions", type=int, default=8_192)
    parser.add_argument("--budgets-mb", default="4,8,16,32")
    parser.add_argument("--build-root", type=Path, default=None)
    parser.add_argument("--keep-build", action="store_true")
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--evict-after-read", action="store_true")
    args = parser.parse_args()

    data = load_sessions(args.data)
    n_sessions = min(args.n_sessions, len(data.train))
    sessions = data.train[:n_sessions]
    step = max(1, len(data.test) // args.n_query)
    query_indices = list(range(0, len(data.test), step))[:args.n_query]
    queries = serving_prefixes([data.test[i] for i in query_indices])

    exact = ExactJaccard()
    for session in sessions:
        exact.insert(session)
    exact.finalize()
    truth = [exact.topk(query, 10) for query in queries]

    rng = np.random.default_rng(20261029)
    profiles = rng.normal(0.0, 1.0,
                          size=(n_sessions, 64)).astype(np.float32)
    root = (args.build_root or
            Path("/tmp") / f"skc_bounded_tier_{args.data.stem}")
    if root.exists():
        shutil.rmtree(root)
    build_started = time.perf_counter()
    store = build_bounded_tier(
        sessions, profiles, root, block_sessions=args.block_sessions,
        bands=32, rows=2, hot_blocks=1,
    )
    build_seconds = time.perf_counter() - build_started
    cold_bytes = sum(path.stat().st_size for path in root.rglob("*")
                     if path.is_file())

    rows = []
    for budget_mb in [float(x) for x in args.budgets_mb.split(",")]:
        capped = BoundedTieredPostingStore(
            root, memory_budget_bytes=int(budget_mb * 1e6),
            bands=32, rows=2, k=10, max_candidates=512,
            direct_io=args.direct_io,
            evict_after_read=args.evict_after_read,
        )
        recalls, latencies = [], []
        resident_max = 0
        for query, ground_truth in zip(queries, truth):
            t0 = time.perf_counter()
            pred, _ = capped.search(query, return_profiles=True)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            recalls.append(recall_at_k(pred, ground_truth, 10))
            resident_max = max(resident_max, capped.resident_bytes)
        row = {
            "budget_mb": budget_mb,
            "recall_at_10": float(np.mean(recalls)),
            "lookup_p50_ms": percentile(latencies, 50),
            "lookup_p90_ms": percentile(latencies, 90),
            "lookup_p99_ms": percentile(latencies, 99),
            "qps": len(queries) / max(sum(latencies) / 1000.0, 1e-9),
            "resident_max_mb": resident_max / 1e6,
            "resident_end_mb": capped.resident_bytes / 1e6,
            "hot_blocks": len(capped._hot),
            "disk_bucket_reads": capped.disk_bucket_reads,
            "disk_item_reads": capped.disk_item_reads,
            "disk_index_reads": capped.disk_index_reads,
            "disk_bucket_bytes_mb": capped.disk_bucket_bytes / 1e6,
            "disk_item_bytes_mb": capped.disk_item_bytes / 1e6,
            "direct_io": bool(args.direct_io),
            "evict_after_read": bool(args.evict_after_read),
            "temporary_bytes_max_mb": capped.temporary_bytes_max / 1e6,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    result = {
        "dataset": args.data.stem,
        "n_sessions": n_sessions,
        "n_queries": len(queries),
        "block_sessions": args.block_sessions,
        "blocks": len(store.block_ids),
        "build_seconds": build_seconds,
        "cold_storage_bytes": cold_bytes,
        "cold_storage_mb": cold_bytes / 1e6,
        "protocol": ("hard hot-memory cap; sparse on-disk bucket indexes; "
                     "cold item records read with bounded temporary arrays"),
        "budgets": rows,
    }
    dest = Path("results/systems") / f"bounded_tier_{args.data.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(f"wrote {dest}")
    if not args.keep_build and args.build_root is None:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
