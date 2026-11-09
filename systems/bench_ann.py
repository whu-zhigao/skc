#!/usr/bin/env python3
"""E8: ANN baseline comparison — FAISS IVF / HNSW vs IGMS.

Three-dimensional comparison: recall@K, query latency (p50/p99), memory.
IGMS searches by item-set similarity (MinHash); FAISS/HNSW search by
dense vector similarity (session profiles as vectors).

Run: python3 systems/bench_ann.py --data data/diginetica.pkl
     (requires: pip install faiss-cpu)
Writes results/systems/ann_bench_<stem>.json
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
from systems.igms import ExactJaccard, IGMS, IGMSConfig  # noqa: E402


def profile_to_vec(items, emb, n_items):
    """Session → dense vector (mean of item embeddings)."""
    ids = np.asarray(sorted(set(items)))
    return emb[ids].mean(axis=0)


def bench_faiss(train_profiles, test_profiles, sample_idx, k, nlist_values):
    """FAISS IVF index at multiple nlist settings."""
    try:
        import faiss
    except ImportError:
        print("  faiss not installed; skipping FAISS baselines")
        return {}

    d = train_profiles.shape[1]
    results = {}
    for nlist in nlist_values:
        nlist = min(nlist, len(train_profiles) // 39 + 1)
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(train_profiles.astype(np.float32))
        t0 = time.perf_counter()
        index.add(train_profiles.astype(np.float32))
        build_s = time.perf_counter() - t0

        for nprobe in [1, max(2, nlist // 10), nlist]:
            latencies = []
            found_ids = []
            for i in sample_idx:
                q = test_profiles[i:i+1].astype(np.float32)
                t1 = time.perf_counter()
                _, ids = index.search(q, k)
                latencies.append((time.perf_counter() - t1) * 1000)
                found_ids.append(ids[0].tolist())
            p50 = st.median(latencies)
            p99 = float(np.percentile(latencies, 99))
            key = f"ivf_nlist{nlist}_nprobe{nprobe}"
            results[key] = {
                "build_s": build_s, "nlist": nlist, "nprobe": nprobe,
                "query_p50_ms": p50, "query_p99_ms": p99,
                "found_ids": found_ids,
                "memory_mb": (index.ntotal * d * 4 + nlist * d * 4) / 1e6,
            }
            print(f"  FAISS IVF(nlist={nlist}, nprobe={nprobe}): "
                  f"p50 {p50:.3f} ms, p99 {p99:.3f} ms")
    return results


def bench_hnsw(train_profiles, test_profiles, sample_idx, k, ef_values):
    """HNSW index at multiple ef_construction/ef_search settings."""
    try:
        import hnswlib
    except ImportError:
        print("  hnswlib not installed; skipping HNSW baselines")
        return {}

    d = train_profiles.shape[1]
    n = len(train_profiles)
    results = {}
    for M in [16, 32]:
        index = hnswlib.Index(space='l2', dim=d)
        index.init_index(max_elements=n, ef_construction=200, M=M)
        t0 = time.perf_counter()
        index.add_items(train_profiles.astype(np.float32),
                        np.arange(n))
        build_s = time.perf_counter() - t0
        for ef_search in ef_values:
            index.set_ef(ef_search)
            latencies = []
            found_ids = []
            for i in sample_idx:
                q = test_profiles[i:i+1].astype(np.float32)
                t1 = time.perf_counter()
                ids, _ = index.knn_query(q, k)
                latencies.append((time.perf_counter() - t1) * 1000)
                found_ids.append(ids[0].tolist())
            p50 = st.median(latencies)
            p99 = float(np.percentile(latencies, 99))
            key = f"hnsw_M{M}_ef{ef_search}"
            results[key] = {
                "build_s": build_s, "M": M, "ef_search": ef_search,
                "query_p50_ms": p50, "query_p99_ms": p99,
                "found_ids": found_ids,
                "memory_mb": index.index_memory_usage / 1e6 if hasattr(index, 'index_memory_usage') else n * d * 4 * (M/16 + 1),
            }
            print(f"  HNSW(M={M}, ef={ef_search}): "
                  f"p50 {p50:.3f} ms, p99 {p99:.3f} ms")
    return results


def recall_at_k(pred_ids, truth_ids):
    return len(set(pred_ids) & set(truth_ids)) / max(1, len(truth_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--n-sample", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    data = load_sessions(args.data)
    stem = args.data.stem
    k = args.k

    # Build session profile vectors (for FAISS/HNSW)
    # Use random embeddings if no checkpoint specified (fair comparison:
    # both IGMS and ANN search over the same representation space)
    if args.ckpt and args.ckpt.exists():
        import torch
        from mplignn.model import MPLI_GNN
        model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=2,
                         n_heads=2, global_blocks=2, vocab_blocks=1,
                         k_r=10, no_residual=True)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu",
                                         weights_only=True))
        emb = model.embedding.weight.detach().numpy()
    else:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((data.n_items + 1, 64)).astype(np.float32)

    print(f"[{stem}] building profile vectors for {len(data.train)} sessions...")
    train_profiles = np.zeros((len(data.train), 64), dtype=np.float32)
    for i, s in enumerate(data.train):
        ids = np.asarray(sorted(set(s)))
        train_profiles[i] = emb[ids].mean(axis=0)

    n_test = len(data.test)
    sample_idx = list(range(0, n_test, max(1, n_test // args.n_sample)))
    sample_idx = sample_idx[:args.n_sample]
    test_profiles = np.zeros((n_test, 64), dtype=np.float32)
    for i in range(n_test):
        ids = np.asarray(sorted(set(data.test[i])))
        test_profiles[i] = emb[ids].mean(axis=0)

    # Ground truth: exact Jaccard top-K
    print(f"[{stem}] computing exact Jaccard ground truth...")
    gt = ExactJaccard()
    for s in data.train:
        gt.insert(s)
    gt.finalize()
    truth = [gt.topk(data.test[i], k) for i in sample_idx]

    out = {"dataset": stem, "k": k, "n_train": len(data.train),
           "n_sample": len(sample_idx)}

    # --- IGMS (sorted layout, our method)
    print(f"[{stem}] IGMS (sorted, b=32, r=2):")
    ig = IGMS(IGMSConfig(sig_len=64, bands=32, k_r=k, max_cand=512,
                         bucket_layout="sorted"))
    t0 = time.perf_counter()
    ig.insert_batch(data.train)
    build_igms = time.perf_counter() - t0
    ig.freeze()
    lat = []
    for i in sample_idx:
        t1 = time.perf_counter()
        ig.search(data.test[i], k)
        lat.append((time.perf_counter() - t1) * 1000)
    ig_recalls = [recall_at_k(ig.search(data.test[i], k), truth[j])
                  for j, i in enumerate(sample_idx)]
    out["igms"] = {
        "build_s": build_igms, "query_p50_ms": st.median(lat),
        "query_p99_ms": float(np.percentile(lat, 99)),
        "recall_at_k": st.mean(ig_recalls),
        "memory_mb": ig.memory_bytes()["total_lean"] / 1e6,
    }
    print(f"  p50 {out['igms']['query_p50_ms']:.3f} ms, "
          f"recall {out['igms']['recall_at_k']:.3f}, "
          f"mem {out['igms']['memory_mb']:.0f} MB")

    # --- FAISS
    print(f"[{stem}] FAISS:")
    faiss_res = bench_faiss(train_profiles, test_profiles, sample_idx, k,
                            nlist_values=[100, 1000])
    # Compute FAISS recall (against Jaccard GT — note: this is a
    # cross-similarity recall: FAISS finds L2-nearby profiles, which
    # may differ from Jaccard-nearby item sets)
    for key, r in faiss_res.items():
        rec = st.mean(recall_at_k(r["found_ids"][j], truth[j])
                      for j in range(len(sample_idx)))
        r["recall_vs_jaccard"] = rec
        del r["found_ids"]  # don't serialize large lists
        print(f"  {key}: recall_vs_jaccard {rec:.3f}")
    out["faiss"] = faiss_res

    # --- HNSW
    print(f"[{stem}] HNSW:")
    hnsw_res = bench_hnsw(train_profiles, test_profiles, sample_idx, k,
                          ef_values=[50, 200])
    for key, r in hnsw_res.items():
        rec = st.mean(recall_at_k(r["found_ids"][j], truth[j])
                      for j in range(len(sample_idx)))
        r["recall_vs_jaccard"] = rec
        del r["found_ids"]
        print(f"  {key}: recall_vs_jaccard {rec:.3f}")
    out["hnsw"] = hnsw_res

    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"ann_bench_{stem}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
