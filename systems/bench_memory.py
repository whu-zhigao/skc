#!/usr/bin/env python3
"""Systems benchmark E7: tracemalloc-verified memory footprints.

Accounted bytes (memory_bytes()) vs measured Python heap (tracemalloc)
for: IGMS in each bucket layout, and the reference neighbor-item store.
Run: python3 systems/bench_memory.py --data data/diginetica.pkl
"""
import argparse
import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from mplignn.data import load_sessions  # noqa: E402
from systems.igms import IGMS, IGMSConfig  # noqa: E402


def measured_mb(fn):
    tracemalloc.start()
    obj = fn()
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return obj, cur / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    data = load_sessions(args.data)
    stem = args.data.stem
    out = {"dataset": stem, "n_train": len(data.train)}

    def build(layout, values=False):
        cfg = IGMSConfig(sig_len=64, bands=32, k_r=10, max_cand=512,
                         bucket_layout=layout, keep_item_sets=False)
        ig = IGMS(cfg)
        pooled = None
        if values:
            rng = np.random.default_rng(0)
            pooled = rng.standard_normal((len(data.train), 64),
                                         dtype=np.float32)
        ig.insert_batch(data.train, pooled=pooled)
        ig.freeze()
        return ig

    for layout in ["dict", "sorted", "packed"]:
        ig, mb = measured_mb(lambda: build(layout, values=True))
        acc = ig.memory_bytes()
        out[layout] = {"measured_heap_mb": mb,
                       "accounted_mb": acc["total_lean"] / 1e6,
                       "breakdown_mb": {k: v / 1e6
                                        for k, v in acc.items()
                                        if k != "total"}}
        print(f"[{stem}] {layout:7s} measured {mb:7.1f} MB | accounted "
              f"{acc['total_lean']/1e6:7.1f} MB "
              f"(sig {acc['signatures']/1e6:.0f} + buckets "
              f"{acc['buckets']/1e6:.0f} + values {acc['values']/1e6:.0f})")

    def ref_store():
        m, k, l = len(data.train), 10, 12
        return torch_like(m, k, l)

    def torch_like(m, k, l):
        return np.zeros((m, k, l), dtype=np.int64)

    _, mb = measured_mb(ref_store)
    out["reference_neighbor_store"] = {"measured_heap_mb": mb}
    print(f"[{stem}] reference int64 neighbor store (train side): "
          f"{mb:.1f} MB")

    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"memory_{stem}.json"
    f.write_text(json.dumps(out, indent=2))
    print("wrote", f)


if __name__ == "__main__":
    main()
