#!/usr/bin/env python3
"""Reader/publisher scaling test for snapshot-pinned page publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from typing import Dict, List

import numpy as np

from .page_consistency import PageDirtyError, PageLocalDispersionStore
from .source_snapshot import pin_array


def pct(values: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def run_case(n: int, dimension: int, page_size: int, readers: int,
             dirty_items: int, refresh_delay_ms: float, cycles: int,
             seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    sessions = [[int((i * 37 + j * 101) % 512)] for i in range(n) for j in range(1)]
    source = np.zeros((n, dimension), dtype=np.float32)
    store = PageLocalDispersionStore(sessions, source, page_size=page_size, theta=0.1)
    stop = threading.Event()
    lock = threading.Lock()
    acq_ms: List[float] = []
    pub_ms: List[float] = []
    counters = {"clean_reads": 0, "soft_misses": 0, "publications": 0, "errors": 0}

    def reader() -> None:
        local_rng = np.random.default_rng(seed + 1000)
        while not stop.is_set():
            ids = local_rng.integers(0, n, size=320).tolist()
            start = time.perf_counter_ns()
            try:
                views, epoch = store.snapshot_versioned(ids)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                with lock:
                    acq_ms.append(elapsed)
                if any(v.dirty for v in views):
                    with lock:
                        counters["soft_misses"] += 1
                    continue
                # Immutable page arrays and epoch are acquired in one critical
                # section.  Any non-dirty view must obey its reported bound.
                if any(v.dispersion_bound > 0.1 + 1e-6 for v in views):
                    raise AssertionError("clean page exceeded dispersion bound")
                with lock:
                    counters["clean_reads"] += 1
            except PageDirtyError:
                with lock:
                    counters["soft_misses"] += 1
            except Exception:
                with lock:
                    counters["errors"] += 1
                raise

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(readers)]
    for thread in threads:
        thread.start()
    hot_pool = np.arange(512)
    for cycle in range(cycles):
        touched = rng.choice(hot_pool, size=dirty_items, replace=False)
        drift = {int(item): 0.2 for item in touched}
        source[:] = float(cycle + 1)
        snapshot = pin_array(source, version=cycle + 1)
        store.update_drift(drift)
        if refresh_delay_ms:
            time.sleep(refresh_delay_ms / 1e3)
        start = time.perf_counter_ns()
        published = store.refresh(snapshot)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        with lock:
            pub_ms.append(elapsed)
            counters["publications"] += published
    stop.set()
    for thread in threads:
        thread.join(timeout=5)
    return {
        "readers": readers,
        "dirty_items": dirty_items,
        "refresh_delay_ms": refresh_delay_ms,
        "cycles": cycles,
        **counters,
        "acquisition_ms": {"p50": pct(acq_ms, 50), "p99": pct(acq_ms, 99)},
        "publication_ms": {"p50": pct(pub_ms, 50), "p99": pct(pub_ms, 99)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cases = []
    for readers in (1, 2, 4, 8):
        cases.append(run_case(
            n=4096, dimension=32, page_size=128, readers=readers,
            dirty_items=80, refresh_delay_ms=2.0, cycles=20, seed=1707,
        ))
    for dirty_items in (20, 80, 320):
        cases.append(run_case(
            n=4096, dimension=32, page_size=128, readers=4,
            dirty_items=dirty_items, refresh_delay_ms=2.0, cycles=20,
            seed=1707 + dirty_items,
        ))
    result = {
        "protocol": "CPU threads; atomic snapshot_versioned acquisition; immutable source snapshot; dirty-page soft miss",
        "cases": cases,
        "summary": {
            "clean_reads": sum(x["clean_reads"] for x in cases),
            "soft_misses": sum(x["soft_misses"] for x in cases),
            "publications": sum(x["publications"] for x in cases),
            "contract_violations": sum(x["errors"] for x in cases),
        },
    }
    out = args.out or (Path(__file__).resolve().parents[1] / "results" / "systems" / "publication_matrix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(out)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
