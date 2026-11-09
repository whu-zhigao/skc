#!/usr/bin/env python3
"""Measure candidate search while posting blocks are inserted and frozen."""
import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mplignn.data import load_sessions  # noqa: E402
from systems.igms import IGMS, IGMSConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--block-sessions", type=int, default=32768)
    ap.add_argument("--reader-threads", type=int, default=2)
    args = ap.parse_args()
    data = load_sessions(args.data)
    ig = IGMS(IGMSConfig(
        sig_len=64, bands=32, k_r=10, bucket_layout="sorted",
        keep_item_sets=True, compact_item_sets=True,
        discard_signatures=True,
    ))
    queries = [q for q in data.test if len(q) >= 2][:200]
    stop = threading.Event()
    latencies = []
    errors = []
    freeze_times = []

    def reader():
        local = []
        while not stop.is_set():
            query = queries[len(local) % len(queries)]
            t0 = time.perf_counter()
            try:
                ig.search(query, 10)
                local.append((time.perf_counter() - t0) * 1000)
            except Exception as exc:
                errors.append(repr(exc))
        latencies.extend(local)

    threads = [threading.Thread(target=reader, daemon=True)
               for _ in range(args.reader_threads)]
    for thread in threads:
        thread.start()
    for start in range(0, len(data.train), args.block_sessions):
        stop_index = min(start + args.block_sessions, len(data.train))
        ig.insert_batch(data.train[start:stop_index])
        t0 = time.perf_counter()
        ig.freeze()
        freeze_times.append((time.perf_counter() - t0) * 1000)
    stop.set()
    for thread in threads:
        thread.join(timeout=5)

    out = {
        "dataset": args.data.stem,
        "n_sessions": len(data.train),
        "n_blocks": len(ig.posting_blocks),
        "reader_threads": args.reader_threads,
        "searches": len(latencies),
        "search_p50_ms": statistics.median(latencies),
        "search_p99_ms": float(np.percentile(latencies, 99)),
        "freeze_p50_ms": statistics.median(freeze_times),
        "reader_errors": errors,
    }
    dest = Path("results/systems") / (
        f"concurrent_posting_{args.data.stem}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
