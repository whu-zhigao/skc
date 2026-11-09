#!/usr/bin/env python3
"""Systems benchmark E1+E2: incremental index vs static rebuild.

E1 (static regime)  -- build cost, per-query latency (p50/p99), recall@K
                       against full-scan exact Jaccard, memory footprint:
                       FastRelationIndex (baseline, static heuristic) vs
                       IGMS (MinHash-band store).
E2 (streaming regime) -- corpus arrives in 10 chunks; IGMS absorbs the
                       stream incrementally, the baseline must rebuild.
                       Reports amortized cost per chunk and recall drift.

Run:  python3 systems/bench_retrieval.py --data data/diginetica.pkl
      python3 systems/bench_retrieval.py --data data/gowalla.pkl
Writes results/systems/retrieval_bench_<stem>.json
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from mplignn.data import FastRelationIndex, load_sessions  # noqa: E402
from systems.igms import ExactJaccard, IGMS, IGMSConfig  # noqa: E402


def baseline_neighbor_ids(anchor, corpus, k=10, l_cap=12, max_cand=64):
    """The reference heuristic (FastRelationIndex search logic), returning
    neighbor SESSION ids so recall is computable. Identical parameters."""
    inv = {}
    for idx, s in enumerate(corpus):
        for i in set(s):
            inv.setdefault(i, []).append(idx)
    freq = {i: len(v) for i, v in inv.items()}
    out = []
    for s in anchor:
        items = sorted(set(s), key=lambda i: freq.get(i, 1 << 30))
        cands = {}
        for it in items[:3]:
            for j in inv.get(it, [])[:max_cand]:
                if j != idx:
                    cands[j] = cands.get(j, 0) + 1
        top = sorted(cands.items(), key=lambda kv: -kv[1])[:k]
        ids = [j for j, _ in top]
        while len(ids) < k:
            ids.append(0)
        out.append(ids)
    return out


def recall_at_k(pred: list, truth: list) -> float:
    p, t = set(pred), set(truth)
    return len(p & t) / max(1, len(t))


def bench_static(data, stem, out):
    n_test, n_train = len(data.test), len(data.train)
    sample = list(range(0, n_test, max(1, n_test // 300)))[:300]
    k = 10

    # ---- ground truth: exact Jaccard over the full corpus
    t0 = time.perf_counter()
    gt = ExactJaccard()
    for s in data.train:
        gt.insert(s)
    gt.finalize()
    gt_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    truth = [gt.topk(data.test[i], k) for i in sample]
    gt_query_ms = 1000 * (time.perf_counter() - t0) / len(sample)

    # ---- baseline: FastRelationIndex, forced rebuild (no cache read)
    cache = Path("results/systems") / f"force_{stem}_train.npz"
    t0 = time.perf_counter()
    base = FastRelationIndex(data.train, data.train, data.n_items, k,
                             cache=None)
    base_build = time.perf_counter() - t0
    base_mem = base.neigh_items.element_size() * base.neigh_items.nelement()
    # baseline recall: its stored neighbors for the TRAIN anchors are the
    # same ids IGMS serves; map test anchors through its own interface by
    # rebuilding over test anchors
    t0 = time.perf_counter()
    base_test = FastRelationIndex(data.test, data.train, data.n_items, k,
                                  cache=None)
    base_test_build = time.perf_counter() - t0
    base_mem_total = base_mem + base_test.neigh_items.element_size() * \
        base_test.neigh_items.nelement()
    base_ids = baseline_neighbor_ids([data.test[i][:-1] if len(data.test[i]) > 1 else data.test[i] for i in sample],
                                     data.train, k)
    base_recall = st.mean(
        recall_at_k(base_ids[j], truth[j]) for j in range(len(sample)))

    # ---- IGMS: streaming build + band search (knob sweep over rows/bands)
    cfg = IGMSConfig(sig_len=64, bands=32, k_r=k, max_cand=512, bucket_layout='packed')  # r=2
    ig = IGMS(cfg)
    t0 = time.perf_counter()
    ig.insert_batch(data.train)
    ig_build = time.perf_counter() - t0
    lat = []
    t0 = time.perf_counter()
    for i in sample:
        t1 = time.perf_counter()
        ig.search(data.test[i], k)
        lat.append((time.perf_counter() - t1) * 1000)
    ig_query_total = time.perf_counter() - t0
    ig_p50, ig_p99 = st.median(lat), np.percentile(lat, 99)
    ig_recall = st.mean(
        recall_at_k(ig.search(data.test[i], k), truth[j])
        for j, i in enumerate(sample))
    mem = ig.memory_bytes()

    res = {
        "dataset": stem, "n_train": n_train, "n_test": n_test,
        "k": k,
        "ground_truth": {"build_s": gt_build,
                         "query_ms_per_anchor": gt_query_ms},
        "baseline_static": {
            "train_build_s": base_build, "test_build_s": base_test_build,
            "rebuild_needed_per_corpus_change": True,
            "train_mem_mb": base_mem / 1e6, "total_mem_mb": base_mem_total / 1e6,
            "recall_at_10": base_recall,
            "query_ms_per_anchor": 0.0,
            "note": "neighbors precomputed at build; query is a gather"},
        "igms": {
            "insert_total_s": ig_build,
            "insert_us_per_session": 1e6 * ig_build / n_train,
            "query_p50_ms": ig_p50, "query_p99_ms": ig_p99,
            "query_ms_per_anchor": 1000 * ig_query_total / len(sample),
            "recall_at_10": ig_recall,
            "candidates_per_query": ig.probe_ops / len(sample),
            "mem_mb": {kk: vv / 1e6 for kk, vv in mem.items()},
        },
    }
    print(f"[{stem}] exact-GT query {gt_query_ms:.2f} ms | baseline build "
          f"{base_build:.1f}+{base_test_build:.1f}s recall {base_recall:.3f} "
          f"mem {base_mem_total/1e6:.0f}MB")
    print(f"[{stem}] IGMS insert {1e6*ig_build/n_train:.0f} us/sess, query "
          f"p50 {ig_p50:.2f} p99 {ig_p99:.2f} ms, recall {ig_recall:.3f}, "
          f"mem {mem['total']/1e6:.0f}MB")
    out[stem] = res


def bench_streaming(data, stem, out, chunks=10):
    n_train = len(data.train)
    k = 10
    sample = list(range(0, len(data.test), max(1, len(data.test) // 150)))[:150]

    ig = IGMS(IGMSConfig(sig_len=64, bands=32, k_r=k, max_cand=512, bucket_layout='packed'))
    base_time, ig_time = 0.0, 0.0
    checkpoints = []
    step = max(1, n_train // chunks)
    for c in range(1, chunks + 1):
        upto = min(n_train, c * step)
        t0 = time.perf_counter()
        ig.insert_batch(data.train[(c - 1) * step:upto])
        ig_chunk = time.perf_counter() - t0
        ig_time += ig_chunk
        # baseline strategy: rebuild its index over the same prefix
        t0 = time.perf_counter()
        FastRelationIndex(data.train[:upto], data.test,
                          data.n_items, k, cache=None)
        base_chunk = time.perf_counter() - t0
        base_time += base_chunk
        # recall against exact over the same prefix
        gt = ExactJaccard()
        for s in data.train[:upto]:
            gt.insert(s)
        gt.finalize()
        truth = [gt.topk(data.test[i], k) for i in sample]
        rec = st.mean(recall_at_k(ig.search(data.test[i], k), truth[j])
                      for j, i in enumerate(sample))
        checkpoints.append({
            "corpus": upto, "igms_chunk_s": ig_chunk,
            "baseline_rebuild_s": base_chunk, "igms_cumulative_s": ig_time,
            "baseline_cumulative_s": base_time, "igms_recall": rec})
    out[stem + "_stream"] = {
        "dataset": stem, "chunks": checkpoints,
        "igms_total_s": ig_time, "baseline_total_s": base_time,
        "speedup": base_time / max(ig_time, 1e-9)}
    print(f"[{stem}] streaming {chunks} chunks: IGMS {ig_time:.1f}s vs "
          f"rebuilds {base_time:.1f}s ({base_time/max(ig_time,1e-9):.1f}x); "
          f"final recall {checkpoints[-1]['igms_recall']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--skip-stream", action="store_true")
    args = ap.parse_args()
    data = load_sessions(args.data)
    stem = args.data.stem
    out = {}
    bench_static(data, stem, out)
    if not args.skip_stream:
        bench_streaming(data, stem, out)
    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"retrieval_bench_{stem}.json"
    f.write_text(json.dumps(out, indent=2))
    print("wrote", f)


if __name__ == "__main__":
    main()
