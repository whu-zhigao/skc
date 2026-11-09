#!/usr/bin/env python3
"""Systems benchmark E5: bucket-layout operating points + band sweep.

Layouts (dict / sorted / packed) at fixed bands=32, then a band sweep
(b in {8,16,32,64} at r=2) with the sorted layout: insert cost, query
p50/p99, recall@10 vs full-scan exact, footprint.

Run: python3 systems/bench_layouts.py --data data/diginetica.pkl
Writes results/systems/layout_sweep_<stem>.json
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from mplignn.data import load_sessions  # noqa: E402

def query_prefix(s):
    return s[:-1] if len(s) > 1 else s
from systems.igms import ExactJaccard, IGMS, IGMSConfig  # noqa: E402


def recall_at_k(pred, truth):
    return len(set(pred) & set(truth)) / max(1, len(truth))


def run_cfg(data, sample, truth, k, cfg):
    ig = IGMS(cfg)
    t0 = time.perf_counter()
    ig.insert_batch(data.train)
    build_s = time.perf_counter() - t0
    ig.freeze()
    lat = []
    for i in sample:
        t1 = time.perf_counter()
        ig.search(query_prefix(data.test[i]), k)
        lat.append((time.perf_counter() - t1) * 1000)
    rec = st.mean(recall_at_k(ig.search(query_prefix(data.test[i]), k), truth[j])
                  for j, i in enumerate(sample))
    mem = ig.memory_bytes()
    return {"insert_us_per_session": 1e6 * build_s / len(data.train),
            "query_p50_ms": st.median(lat),
            "query_p99_ms": float(np.percentile(lat, 99)),
            "recall_at_10": rec,
            "footprint_mb": mem["total_lean"] / 1e6,
            "buckets_mb": mem["buckets"] / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    data = load_sessions(args.data)
    stem, k = args.data.stem, 10
    sample = list(range(0, len(data.test), max(1, len(data.test) // 300)))[:300]

    t0 = time.perf_counter()
    gt = ExactJaccard()
    for s in data.train:
        gt.insert(s)
    gt.finalize()
    truth = [gt.topk(query_prefix(data.test[i]), k) for i in sample]
    print(f"[{stem}] ground truth ready ({time.perf_counter()-t0:.0f}s)")

    out = {"dataset": stem, "n_train": len(data.train),
           "layouts_b32": {}, "band_sweep_sorted": {}}
    for layout in ["dict", "sorted", "packed"]:
        cfg = IGMSConfig(sig_len=64, bands=32, k_r=k, max_cand=512,
                         bucket_layout=layout)
        r = run_cfg(data, sample, truth, k, cfg)
        out["layouts_b32"][layout] = r
        print(f"[{stem}] {layout:7s} insert {r['insert_us_per_session']:5.0f}us "
              f"p50 {r['query_p50_ms']:6.2f} p99 {r['query_p99_ms']:6.2f}ms "
              f"recall {r['recall_at_10']:.3f} mem {r['footprint_mb']:.0f}MB")
    for bands in [8, 16, 32, 64]:
        cfg = IGMSConfig(sig_len=bands * 2, bands=bands, k_r=k, max_cand=512,
                         bucket_layout="sorted")
        r = run_cfg(data, sample, truth, k, cfg)
        out["band_sweep_sorted"][f"b{bands}_r2"] = r
        print(f"[{stem}] sorted b={bands:2d} insert "
              f"{r['insert_us_per_session']:5.0f}us p50 "
              f"{r['query_p50_ms']:6.2f}ms recall {r['recall_at_10']:.3f} "
              f"mem {r['footprint_mb']:.0f}MB")
    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"layout_sweep_{stem}.json"
    f.write_text(json.dumps(out, indent=2))
    print("wrote", f)


if __name__ == "__main__":
    main()
