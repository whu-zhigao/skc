#!/usr/bin/env python3
"""C1: Tier-2 item-projection cache. Active-item fraction, stream hit
rate (freq admission vs LRU), and generation-invalidation volume vs
DAI at item level from the A2 trace."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LRUCache, config_out, ensure_results, load_data,
                    write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--trace", type=Path, default=None)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()
    stem = args.data.stem
    out_path, skip = config_out("c1", stem, args.out, args.force)
    if skip:
        print(f"[c1] exists, skipping: {out_path} (--force to rerun)")
        return
    data = load_data(args.data)

    train_items = {i for s in data.train for i in s}
    test_items = [i for s in data.test for i in s]
    active = train_items | set(test_items)
    stream = test_items
    freq = Counter(stream)

    rows = []
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        cap = max(1, int(len(active) * frac))
        top = {i for i, _ in freq.most_common(cap)}
        hit_f = sum(1 for i in stream if i in top) / len(stream)
        lru = LRUCache(cap)
        hit_l = sum(int(lru.request(i)) for i in stream) / len(stream)
        rows.append({"budget_frac": frac, "budget_items": cap,
                     "freq_hit": hit_f, "lru_hit": hit_l,
                     "bytes_fp32": cap * 3 * args.d * 4})
        print(f"  {frac:.2f}: freq {hit_f:.3f} lru {hit_l:.3f}")

    out = {"dataset": stem, "n_items_total": data.n_items,
           "active_items": len(active),
           "active_fraction": len(active) / data.n_items,
           "budgets": rows}
    trace = args.trace or (ensure_results() / f"drift_trace_{stem}.npz")
    if trace.exists():
        drift = np.load(trace)["drift"]
        out["invalidation"] = [
            {"quantile": q,
             "theta": float(np.percentile(drift, q)),
             "dirty_item_frac": float(
                 (drift > np.percentile(drift, q)).mean())}
            for q in (50, 75, 90, 95, 99)]
    write_json(out_path, out)


if __name__ == "__main__":
    main()
