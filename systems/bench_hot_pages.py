#!/usr/bin/env python3
"""Benchmark packed hot pages under a hard resident-memory cap."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_icde.common import neighbor_ids, serving_prefixes  # noqa: E402
from mplignn.data import load_sessions  # noqa: E402
from systems.hot_pages import (BoundedPackedPageStore,  # noqa: E402
                               HotPageDirtyError)


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n-query", type=int, default=3000)
    parser.add_argument("--budgets-mb", default="1,2,5,10")
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--theta", type=float, default=0.1)
    args = parser.parse_args()

    data = load_sessions(args.data)
    nq = min(args.n_query, len(data.test))
    nbrs = neighbor_ids(serving_prefixes(data.test[:nq]),
                        data.train, 10)
    freq = Counter(sid for nb in nbrs for sid in nb)
    rng = np.random.default_rng(20261029)
    source = rng.normal(0.0, 1.0,
                        size=(len(data.train), 64)).astype(np.float32)
    rows = []

    for budget_mb in [float(x) for x in args.budgets_mb.split(",")]:
        store = BoundedPackedPageStore(
            int(budget_mb * 1e6), dimension=64,
            page_size=args.page_size, theta_max=args.theta,
        )
        candidates = [sid for sid, _ in freq.most_common()]
        store.admit(candidates, source, data.train)
        hits, soft_misses = [], []
        for nb in nbrs:
            try:
                _, hit_rate, _ = store.service(nb, source)
                hits.append(hit_rate)
            except HotPageDirtyError:
                soft_misses.append(1)
                hits.append(0.0)
        # Stress one maintenance update and refresh all dirty packed pages.
        t0 = time.perf_counter()
        store.update_drift({int(i): args.theta * 1.5
                            for i in rng.choice(30000, size=80,
                                                replace=False)})
        refresh_ms = (time.perf_counter() - t0) * 1000.0
        dirty = len(store.dirty_pages())
        t0 = time.perf_counter()
        store.refresh(source)
        refresh_ms += (time.perf_counter() - t0) * 1000.0
        row = {
            "budget_mb": budget_mb,
            "admitted_sessions": (0 if store.resident_sids is None
                                  else len(store.resident_sids)),
            "packed_pages": len(store.pages),
            "resident_mb": store.resident_bytes() / 1e6,
            "hard_cap_mb": budget_mb,
            "cap_respected": store.resident_bytes() <= store.memory_budget_bytes,
            "hit_rate": float(np.mean(hits)),
            "soft_miss_fraction": (float(np.mean(soft_misses))
                                   if soft_misses else 0.0),
            "dirty_pages_after_update": dirty,
            "update_plus_refresh_ms": refresh_ms,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    result = {
        "dataset": args.data.stem,
        "n_queries": nq,
        "page_size": args.page_size,
        "dimension": 64,
        "protocol": ("packed admitted rows; natural-id pages are not used for "
                     "hot value placement"),
        "budgets": rows,
    }
    dest = Path("results/systems") / f"packed_hot_pages_{args.data.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
