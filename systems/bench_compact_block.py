#!/usr/bin/env python3
"""Measure the compact frozen-block IGMS layout.

This benchmark separates model payload from index metadata.  It builds a
sorted posting block, converts Python item sets to CSR arrays, discards the
signatures after freeze, and measures the retained heap.  Retrieval recall is
compared with exact full-corpus Jaccard top-k on a systematic query sample.
"""
import argparse
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mplignn.data import load_sessions  # noqa: E402

def query_prefix(s):
    return s[:-1] if len(s) > 1 else s
from systems.igms import ExactJaccard, IGMS, IGMSConfig  # noqa: E402


def recall_at_k(pred, truth, k):
    return len(set(pred) & set(truth)) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--bands", type=int, default=32)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-query", type=int, default=300)
    args = ap.parse_args()

    data = load_sessions(args.data)
    sample = list(range(0, len(data.test),
                        max(1, len(data.test) // args.n_query)))[:args.n_query]
    queries = [query_prefix(data.test[i]) for i in sample]

    truth_index = ExactJaccard()
    for session in data.train:
        truth_index.insert(session)
    truth_index.finalize()
    truth = [truth_index.topk(q, args.k) for q in queries]

    tracemalloc.start()
    index = IGMS(IGMSConfig(
        sig_len=args.bands * args.rows, bands=args.bands, k_r=args.k,
        bucket_layout="sorted", keep_item_sets=True,
        compact_item_sets=True, discard_signatures=True,
    ))
    index.insert_batch(data.train)
    index.freeze()
    index_heap_mb, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latency_ms, recalls = [], []
    for query, ground_truth in zip(queries, truth):
        start = time.perf_counter()
        pred = index.search(query, args.k)
        latency_ms.append((time.perf_counter() - start) * 1000)
        recalls.append(recall_at_k(pred, ground_truth, args.k))

    # Payload allocation is measured separately because the compact index in
    # this benchmark intentionally stores no profile values.
    tracemalloc.start()
    active_ids = np.asarray(sorted({sid for top in truth
                                    for sid in top}), dtype=np.int64)
    codes = np.zeros((len(active_ids), 64), dtype=np.int8)
    scales = np.ones(len(active_ids), dtype=np.float32)
    resident_map = np.full(len(data.train), -1, dtype=np.int32)
    resident_map[active_ids] = np.arange(len(active_ids), dtype=np.int32)
    active_items = np.asarray(sorted({item for query in queries
                                      for item in set(query)}), dtype=np.int64)
    projections = np.zeros((len(active_items), 3, 64), dtype=np.float32)
    sketch = np.zeros((4096, 4), dtype=np.int32)
    payload_heap_mb, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    accounted = index.memory_bytes()
    result = {
        "dataset": args.data.stem,
        "n_sessions": len(data.train),
        "n_queries": len(queries),
        "bands": args.bands,
        "rows": args.rows,
        "index_accounted_mb": accounted["total_lean"] / 1e6,
        "index_measured_heap_mb": index_heap_mb / 1e6,
        "index_breakdown_mb": {
            key: value / 1e6 for key, value in accounted.items()
            if key not in ("total", "total_lean")
        },
        "payload_analytic_mb": (codes.nbytes + scales.nbytes +
                                resident_map.nbytes) / 1e6,
        "active_sessions": len(active_ids),
        "payload_measured_heap_mb": payload_heap_mb / 1e6,
        "active_items": len(active_items),
        "tier2_payload_mb": projections.nbytes / 1e6,
        "sketch_mb": sketch.nbytes / 1e6,
        "recall_at_10_mean": statistics.mean(recalls),
        "lookup_p50_ms": statistics.median(latency_ms),
        "lookup_p99_ms": float(np.percentile(latency_ms, 99)),
    }
    result["analytic_total_mb"] = (
        result["index_accounted_mb"] + result["payload_analytic_mb"] +
        result["tier2_payload_mb"] + result["sketch_mb"])
    result["measured_total_mb"] = (
        result["index_measured_heap_mb"] +
        result["payload_measured_heap_mb"])

    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / f"compact_block_{args.data.stem}_nq{len(queries)}.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("wrote", output)


if __name__ == "__main__":
    main()
