#!/usr/bin/env python3
"""Measure NFA maintenance and cumulative-vs-decayed sketch admission."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mplignn.data import load_sessions  # noqa: E402
from experiments_icde.common import serving_prefixes  # noqa: E402
from systems.bench_retrieval import baseline_neighbor_ids  # noqa: E402


class DecayCountMin:
    def __init__(self, w=4096, d=4, decay=0.70, seed=0):
        rng = np.random.default_rng(seed)
        self.w, self.d, self.decay = w, d, decay
        self.a = rng.integers(1, 1 << 31, size=d, dtype=np.int64)
        self.b = rng.integers(0, 1 << 31, size=d, dtype=np.int64)
        self.table = np.zeros((d, w), dtype=np.float32)
        self._p = (1 << 61) - 1

    def update(self, ids):
        idx = ((ids[:, None] * self.a[None, :] +
                self.b[None, :]) % self._p) % self.w
        for row in range(self.d):
            np.add.at(self.table[row], idx[:, row], 1)

    def age(self):
        self.table *= self.decay

    def estimate(self, ids):
        idx = ((ids[:, None] * self.a[None, :] +
                self.b[None, :]) % self._p) % self.w
        return self.table[np.arange(self.d)[None, :], idx].min(axis=1)


def run(nbrs, capacity, refresh_every, decayed):
    seen = np.empty(0, dtype=np.int64)
    cm = DecayCountMin()
    cached = set()
    hits, refresh_ms = [], []
    for i, nb in enumerate(nbrs):
        cm.update(np.asarray(nb, dtype=np.int64))
        seen = np.unique(np.concatenate(
            (seen, np.asarray(nb, dtype=np.int64))))
        hits.append(sum(sid in cached for sid in nb) / len(nb))
        if len(cached) < capacity:
            cached.update(int(sid) for sid in nb
                          if len(cached) < capacity)
        if (i + 1) % refresh_every == 0 or i == len(nbrs) - 1:
            t0 = time.perf_counter()
            if decayed:
                cm.age()
            order = np.argsort(-cm.estimate(seen))[:capacity]
            cached = set(seen[order].tolist())
            refresh_ms.append((time.perf_counter() - t0) * 1000)
    return np.asarray(hits), refresh_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-query", type=int, default=3000)
    ap.add_argument("--budget-mb", type=float, default=5)
    args = ap.parse_args()
    data = load_sessions(args.data)
    queries = serving_prefixes(data.test[:min(args.n_query,
                                                     len(data.test))])
    nbrs = baseline_neighbor_ids(queries, data.train, 10)
    capacity = int(args.budget_mb * 1e6 / (64 * 4))
    arms = {}
    for decayed in (False, True):
        for window in (64, 256, 1024):
            key = ("decay" if decayed else "cumulative") + f"_W{window}"
            hits, ms = run(nbrs, capacity, window, decayed)
            arms[key] = {
                "hit_rate": float(hits.mean()),
                "refresh_p50_ms": statistics.median(ms),
                "refresh_p99_ms": float(np.percentile(ms, 99)),
                "n_refreshes": len(ms),
            }
    out = {"dataset": args.data.stem, "n_queries": len(nbrs),
           "capacity": capacity, "arms": arms}
    dest = Path("results/systems") / (
        f"admission_maintenance_{args.data.stem}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
