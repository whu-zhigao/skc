#!/usr/bin/env python3
"""E2 (M2): closed-loop PI budget controller under arrival-order drift.

The architecture (Fig. 1) places a PI controller that reallocates the
Tier-1 budget to hold a residency target. This experiment evaluates
it: replay the query stream in arrival order, estimate neighbor
frequencies with a 2-hash Count-Min sketch, and let a PI controller
move the effective budget B_t every window so that the measured
residency (fraction of queries whose full neighbor set is admitted)
tracks the target R*. Baseline: the OPEN-LOOP static budget B_0 frozen
at the first window. Drift is the stream's own arrival-order drift.

Output: results/systems/e2/<ds>_controller_<target>.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import serving_prefixes, config_out, ensure_results, load_data, neighbor_ids, write_json


class CountMin:
    def __init__(self, w=4096, rows=2, seed=0):
        rng = np.random.default_rng(seed)
        self.w, self.rows = w, rows
        self.a = rng.integers(1, w - 1, size=rows)
        self.b = rng.integers(0, w - 1, size=rows)
        self.table = np.zeros((rows, w), dtype=np.float32)

    def _idx(self, x):
        return [(int(x) * int(a) + int(b)) % self.w for a, b in zip(self.a, self.b)]

    def update(self, x, c=1.0):
        for i in self._idx(x):
            self.table[:, i] += c

    def est(self, x):
        return float(min(self.table[r, i] for r, i in enumerate(self._idx(x))))


def hit_rate(admitted, nbrs_win):
    """LOOKUP-level hit rate, the same metric exp_b1's online replay
    uses: each neighbor lookup (flattened stream) hits iff the session
    is admitted. Query-level full-coverage is far stricter and would
    contradict tab:fig:pareto's semantics."""
    h = sum(len(set(nb) & admitted) for nb in nbrs_win)
    t = sum(len(nb) for nb in nbrs_win)
    return h / max(1, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--n-query", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--target", type=float, default=0.90,
                    help="residency target R*")
    ap.add_argument("--b0", type=int, default=1500,
                    help="initial Tier-1 budget in sessions")
    ap.add_argument("--kp", type=float, default=1200.0)
    ap.add_argument("--ki", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    stem = args.data.stem
    name = f"{stem}_controller_R{args.target:g}_nq{args.n_query}"
    out_path, skip = config_out("e2", name, args.out, args.force)
    if skip:
        print(f"[e2] exists, skipping: {out_path}")
        return

    data = load_data(args.data)
    nq = min(args.n_query, len(data.test))
    nbrs_all = [np.asarray(nb) for nb in
                neighbor_ids(serving_prefixes(data.test[:nq]), data.train, args.k)]

    corpus = len(data.train)
    W = args.window
    n_win = nq // W

    def run(closed_loop):
        sketch = CountMin(seed=0)   # fresh sketch per arm
        admitted = set()
        pool = set()                # cumulative candidate pool
        B = args.b0
        integ = 0.0
        series = []
        for w in range(n_win):
            win = range(w * W, (w + 1) * W)
            hits = tot = 0
            # STREAMING within the window, exactly like exp_b1's replay:
            # each lookup is served, THEN observed/admitted, so
            # within-window repeats count as hits (the semantic that
            # produced b1's sketch-NFA numbers).
            for qi in win:
                nb = nbrs_all[qi]
                if len(nb):
                    hits += len(set(nb) & admitted)
                    tot += len(nb)
                for x in nb:
                    sketch.update(int(x))
                    pool.add(int(x))
                if len(pool) > B:
                    admitted = set(sorted(
                        pool, key=lambda v: -sketch.est(v))[:B])
                else:
                    admitted = set(pool)
            R = hits / max(1, tot)
            if closed_loop:
                err = args.target - R
                integ += err
                B = int(np.clip(B + args.kp * err + args.ki * integ,
                                100, corpus))
            series.append({"window": w, "budget": B,
                           "hit_rate": R})
        return series

    open_s = run(closed_loop=False)
    closed_s = run(closed_loop=True)

    def err_stats(s):
        e = np.abs([max(0.0, args.target - x["hit_rate"]) for x in s])
        return {"mean_abs_shortfall": float(e.mean()),
                "p95_abs_shortfall": float(np.percentile(e, 95)),
                "final_budget": s[-1]["budget"],
                "final_hit_rate": s[-1]["hit_rate"]}

    report = {"dataset": stem, "n_query": nq, "window": W,
              "target": args.target, "b0": args.b0, "corpus": corpus,
              "open_loop": {"series": open_s, **err_stats(open_s)},
              "closed_loop": {"series": closed_s, **err_stats(closed_s)}}
    ensure_results()
    write_json(out_path, report)


if __name__ == "__main__":
    main()
