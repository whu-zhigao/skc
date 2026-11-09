#!/usr/bin/env python3
"""Construct a single-protocol SKC resident-memory ledger.

The online arm replays the first --arrival-queries queries in arrival order,
updates a Count--Min sketch, and selects the top --budget-mb sessions.  The
ceiling arm uses exact Jaccard neighbors from a systematic sample.  Both arms
use the same compact full-corpus posting block, so payload and metadata are
never mixed across protocols.
"""
import argparse
import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mplignn.data import load_sessions  # noqa: E402

def query_prefix(s):
    return s[:-1] if len(s) > 1 else s
from systems.bench_retrieval import baseline_neighbor_ids  # noqa: E402
from systems.igms import ExactJaccard, IGMS, IGMSConfig  # noqa: E402
from experiments_icde.common import CountMin  # noqa: E402


def build_index(data, k):
    tracemalloc.start()
    ig = IGMS(IGMSConfig(
        sig_len=64, bands=32, k_r=k, bucket_layout="sorted",
        keep_item_sets=True, compact_item_sets=True,
        discard_signatures=True,
    ))
    ig.insert_batch(data.train)
    ig.freeze()
    heap_mb, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ig, heap_mb / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arrival-queries", type=int, default=3000)
    ap.add_argument("--ceiling-queries", type=int, default=3000)
    ap.add_argument("--budget-mb", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sketch-w", type=int, default=4096)
    ap.add_argument("--sketch-d", type=int, default=4)
    args = ap.parse_args()

    data = load_sessions(args.data)
    ig, index_measured_mb = build_index(data, args.k)
    index_mem = ig.memory_bytes()["total_lean"] / 1e6

    arrival = [s[:-1] if len(s) > 1 else s for s in data.test[:min(args.arrival_queries, len(data.test))]]
    nbrs = baseline_neighbor_ids(arrival, data.train, args.k)
    cm = CountMin(w=args.sketch_w, d=args.sketch_d)
    for nb in nbrs:
        cm.update(nb)
    ids = np.unique([sid for nb in nbrs for sid in nb])
    # B1's admission allowance is expressed in fp32-equivalent profile bytes;
    # the selected rows are then stored as int8 plus an fp32 scale.
    capacity = int(args.budget_mb * 1e6 / (64 * 4))
    cached = set(ids[np.argsort(-cm.estimate(ids))[:capacity]].tolist())
    query_items = sorted({item for q in arrival for item in set(q)})
    online = {
        "protocol": "profiled full-window Count-Min NFA (resident-set accounting; not an online hit-rate claim)",
        "n_queries": len(arrival),
        "budget_mb": args.budget_mb,
        "cached_sessions": len(cached),
        "active_items": len(query_items),
        "tier1_payload_mb": len(cached) * 68 / 1e6,
        "resident_id_map_mb": len(data.train) * 4 / 1e6,
        "tier2_payload_mb": len(query_items) * 3 * 64 * 4 / 1e6,
        "index_mb": index_mem,
        "index_measured_mb": index_measured_mb,
        "sketch_mb": cm.bytes() / 1e6,
    }
    online["total_mb"] = sum(
        online[x] for x in ("tier1_payload_mb", "resident_id_map_mb",
                            "tier2_payload_mb", "index_mb", "sketch_mb"))

    # Measure the online payload objects directly; index heap was measured in
    # the compact-block benchmark, so combine it with this payload allocation.
    tracemalloc.start()
    codes = np.zeros((len(cached), 64), dtype=np.int8)
    scales = np.ones(len(cached), dtype=np.float32)
    resident_map = np.full(len(data.train), -1, dtype=np.int32)
    resident_map[np.fromiter(cached, dtype=np.int32,
                             count=len(cached))] = np.arange(
                                 len(cached), dtype=np.int32)
    projections = np.zeros((len(query_items), 3, 64), dtype=np.float32)
    sketch = np.zeros((args.sketch_d, args.sketch_w), dtype=np.int64)
    payload_heap_mb, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    online["payload_measured_heap_mb"] = payload_heap_mb / 1e6
    online["total_measured_mb"] = (
        payload_heap_mb / 1e6 + index_measured_mb)

    step = max(1, len(data.test) // args.ceiling_queries)
    sample = [s[:-1] if len(s) > 1 else s for s in data.test[::step][:args.ceiling_queries]]
    exact = ExactJaccard()
    for session in data.train:
        exact.insert(session)
    exact.finalize()
    truth = [exact.topk(q, args.k) for q in sample]
    ceiling_sessions = sorted({sid for top in truth for sid in top})
    ceiling_items = sorted({item for q in sample for item in set(q)})
    ceiling = {
        "protocol": "systematic exact-replay active-set ceiling",
        "n_queries": len(sample),
        "cached_sessions": len(ceiling_sessions),
        "active_items": len(ceiling_items),
        "tier1_payload_mb": len(ceiling_sessions) * 68 / 1e6,
        "resident_id_map_mb": len(data.train) * 4 / 1e6,
        "tier2_payload_mb": len(ceiling_items) * 3 * 64 * 4 / 1e6,
        "index_mb": index_mem,
        "sketch_mb": cm.bytes() / 1e6,
    }
    ceiling["total_mb"] = sum(
        ceiling[x] for x in ("tier1_payload_mb", "resident_id_map_mb",
                             "tier2_payload_mb", "index_mb", "sketch_mb"))

    out = {
        "dataset": args.data.stem,
        "n_sessions": len(data.train),
        "posting_blocks": len(ig.posting_blocks),
        "profiled_nfa": online,
        "exact_ceiling": ceiling,
    }
    dest = Path("results/systems") / f"unified_memory_{args.data.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
