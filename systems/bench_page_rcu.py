#!/usr/bin/env python3
"""Request-level gather latency under page-local maintenance.

Readers take immutable page snapshots while a maintainer crosses page
thresholds, holds the dirty window open briefly, and atomically publishes
replacement pages.  Dirty reads become explicit soft misses and fall back to
the source profile matrix.  The benchmark measures the cache gather path, not
the recommendation backbone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mplignn.data import load_sessions  # noqa: E402
from systems.hot_pages import (BoundedPackedPageStore,  # noqa: E402
                               HotPageDirtyError)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n-sessions", type=int, default=50_000)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--requests-per-thread", type=int, default=1_500)
    parser.add_argument("--batch-neighbors", type=int, default=320)
    parser.add_argument("--dirty-window-ms", type=float, default=2.0)
    parser.add_argument("--budget-mb", type=float, default=5.0)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = load_sessions(args.data)
    n = min(args.n_sessions, len(data.train))
    rng = np.random.default_rng(20261029)
    source = rng.normal(0.0, 1.0, size=(n, args.d)).astype(np.float32)
    n_items = max(max(map(int, session)) for session in data.train[:n]) + 1
    store = BoundedPackedPageStore(
        int(args.budget_mb * 1e6), args.d,
        page_size=args.page_size, theta_max=args.theta,
        quantize=args.quantize, quant_bits=8,
    )
    store.admit(range(n), source, data.train[:n])

    start = threading.Event()
    stop = threading.Event()
    latencies: List[float] = []
    soft_misses = [0]
    requests_done = [0]
    errors: List[Exception] = []
    latency_lock = threading.Lock()

    def reader(seed: int) -> None:
        local_rng = np.random.default_rng(seed)
        local_latencies: List[float] = []
        local_soft = 0
        try:
            start.wait()
            for _ in range(args.requests_per_thread):
                ids = local_rng.integers(0, n,
                                         size=args.batch_neighbors).tolist()
                t0 = time.perf_counter()
                try:
                    values, _hit, bound = store.service(ids, source)
                    if bound > args.theta + 1e-12:
                        raise RuntimeError("served beyond dispersion bound")
                except HotPageDirtyError:
                    values = source[np.asarray(ids, dtype=np.int64)]
                    local_soft += 1
                local_latencies.append(time.perf_counter() - t0)
                if values.shape != (args.batch_neighbors, args.d):
                    raise RuntimeError("wrong gather shape")
            with latency_lock:
                latencies.extend(local_latencies)
                soft_misses[0] += local_soft
                requests_done[0] += len(local_latencies)
        except Exception as exc:
            errors.append(exc)
            with latency_lock:
                latencies.extend(local_latencies)
                soft_misses[0] += local_soft
                requests_done[0] += len(local_latencies)

    refresh_times: List[float] = []
    published_pages = [0]

    def maintainer() -> None:
        item_rng = np.random.default_rng(917)
        start.wait()
        while not stop.is_set():
            items = item_rng.choice(n_items, size=80, replace=False)
            store.update_drift({int(i): args.theta * 1.5 for i in items})
            time.sleep(args.dirty_window_ms / 1000.0)
            t0 = time.perf_counter()
            published_pages[0] += store.refresh(source)
            refresh_times.append(time.perf_counter() - t0)
            time.sleep(0.005)

    readers = [threading.Thread(target=reader, args=(1000 + i,))
               for i in range(args.threads)]
    writer = threading.Thread(target=maintainer)
    for thread in readers + [writer]:
        thread.start()
    t0 = time.perf_counter()
    start.set()
    for thread in readers:
        thread.join()
    elapsed = time.perf_counter() - t0
    stop.set()
    writer.join()

    result = {
        "dataset": args.data.stem,
        "protocol": "immutable-page snapshot; dirty-page soft-miss fallback",
        "quantized": bool(args.quantize),
        "resident_mb": store.resident_bytes() / 1e6,
        "resident_cap_mb": args.budget_mb,
        "cap_respected": store.resident_bytes() <= store.memory_budget_bytes,
        "admitted_sessions": (0 if store.resident_sids is None
                              else len(store.resident_sids)),
        "n_sessions": n,
        "dimension": args.d,
        "page_size": args.page_size,
        "theta": args.theta,
        "reader_threads": args.threads,
        "requests_per_thread": args.requests_per_thread,
        "neighbor_rows_per_request_batch": args.batch_neighbors,
        "requests_completed": requests_done[0],
        "soft_misses": soft_misses[0],
        "soft_miss_fraction": soft_misses[0] / max(1, requests_done[0]),
        "gather_latency_ms": {
            "p50": 1000.0 * percentile(latencies, 50),
            "p90": 1000.0 * percentile(latencies, 90),
            "p99": 1000.0 * percentile(latencies, 99),
            "p999": 1000.0 * percentile(latencies, 99.9),
            "max": 1000.0 * max(latencies, default=0.0),
        },
        "refresh_ms": {
            "p50": 1000.0 * percentile(refresh_times, 50),
            "p99": 1000.0 * percentile(refresh_times, 99),
            "max": 1000.0 * max(refresh_times, default=0.0),
        },
        "published_pages": published_pages[0],
        "maintenance_cycles": len(refresh_times),
        "request_gather_qps": requests_done[0] / max(elapsed, 1e-9),
        "wall_seconds": elapsed,
        "contract_violations": len(errors),
        "errors": [repr(exc) for exc in errors],
    }
    out = args.out or (ROOT / "results" / "systems" /
                       f"page_rcu_{args.data.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
