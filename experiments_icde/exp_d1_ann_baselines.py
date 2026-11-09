#!/usr/bin/env python3
"""D1: ANN baselines vs MinHash-LSH vs exact ground truths. FAISS
(IVF + HNSW) on pooled-profile vectors vs a local banded MinHash-LSH
index (stand-in for IGMS; swap in systems.igms.IGMS when its
constructor is confirmed). Ground truths: exact cosine and exact
Jaccard top-k. Skips FAISS if not installed."""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (config_out, jaccard_topk, load_data, load_model,
 serving_prefixes,
                    pool_profiles, write_json)


class MinHashLSH:
    def __init__(self, corpus, r=2, b=64, seed=0):
        rng = np.random.default_rng(seed)
        self.p = (1 << 61) - 1
        self.ha = rng.integers(1, self.p, size=r * b)
        self.hb = rng.integers(0, self.p, size=r * b)
        self.r, self.b = r, b
        self.buckets = [dict() for _ in range(b)]
        self.keys = [self._sig(frozenset(s)) for s in corpus]

    def _sig(self, s: frozenset):
        hs = [int.from_bytes(hashlib.blake2b(
            int(i).to_bytes(8, "little"), digest_size=8).digest(),
            "little") % self.p
            for i in s] or [0]
        rows = (self.ha[:, None] * np.array(hs)[None, :]
                + self.hb[:, None]) % self.p
        return rows.min(axis=1).reshape(self.b, self.r).sum(axis=1)

    def build(self):
        for sid, key in enumerate(self.keys):
            for band, kk in enumerate(key):
                self.buckets[band].setdefault(int(kk), []).append(sid)

    def query(self, s, k=10):
        q = self._sig(frozenset(s))
        cand = set()
        for band, kk in enumerate(q):
            cand.update(self.buckets[band].get(int(kk), ()))
        cand = list(cand)[:10000]
        if not cand:
            return []
        scores = [(sum(int(self.keys[c][b_] == q[b_])
                       for b_ in range(self.b)), c) for c in cand]
        return [c for _, c in sorted(scores, reverse=True)[:k]]


def rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--n-query", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()
    stem = args.data.stem
    tag = f"{stem}_nq{args.n_query}_k{args.k}"
    out_path, skip = config_out("d1", f"{tag}_minhash-lsh", args.out,
                                args.force)
    if skip:
        print(f"[d1] exists, skipping: {out_path} (--force to rerun)")
        return
    data = load_data(args.data)
    model = load_model(data, args.ckpt, device="cpu")
    emb = model.embedding.weight.detach().numpy()
    P = pool_profiles(emb, data.train)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    nq = min(args.n_query, len(data.test))
    queries = serving_prefixes(data.test[:nq])

    import torch
    qv = pool_profiles(emb, queries)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    cos_gt = torch.topk(torch.from_numpy(qv) @ torch.from_numpy(Pn).T,
                        args.k, dim=1).indices.numpy()
    jac_gt = jaccard_topk(queries, data.train, args.k)

    out = {"dataset": stem, "n_query": nq}

    # exact reference row: full-scan exact-Jaccard top-k (the path the
    # suite's neighbor_ids() itself uses) -- recall vs itself is 1.0 by
    # construction; recall_cos = agreement with exact-cosine top-k.
    t0 = time.perf_counter()
    exact_pred = jaccard_topk(queries, data.train, args.k)
    t_exact = time.perf_counter() - t0
    out["exact"] = {
        "build_s": 0.0, "query_s": t_exact,
        "queries_per_s": nq / max(t_exact, 1e-9),
        "deterministic_replay_ok": bool(
            all(set(a) == set(b) for a, b in zip(exact_pred, jac_gt))),
        "recall_jaccard": 1.0,
        "recall_cos": float(np.mean([len(set(p) & set(g)) / args.k
                                     for p, g in zip(exact_pred, cos_gt)])),
        "rss_mb_after_build": rss_mb(),
        "note": "exact full-scan Jaccard top-k -- reference upper "
                "bound; query_s is an independent re-scan timing"}
    t0 = time.perf_counter()
    lsh = MinHashLSH(data.train)
    lsh.build()
    pred = [lsh.query(s, args.k) for s in queries]
    out["minhash_lsh"] = {
        "build_s": time.perf_counter() - t0,
        "recall_cos": float(np.mean([len(set(p) & set(g)) / args.k
                                     for p, g in zip(pred, cos_gt)])),
        "recall_jaccard": float(np.mean([len(set(p) & set(g)) / args.k
                                         for p, g in zip(pred, jac_gt)])),
        "rss_mb_after_build": rss_mb()}

    try:
        import faiss
        d = P.shape[1]
        t0 = time.perf_counter()
        quant = faiss.IndexFlatIP(d)
        ivf = faiss.IndexIVFFlat(quant, d, 256)
        ivf.train(Pn.astype(np.float32))
        ivf.add(Pn.astype(np.float32))
        t_ivf = time.perf_counter() - t0
        t0 = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(d, 32)
        hnsw.add(Pn.astype(np.float32))
        t_hnsw = time.perf_counter() - t0
        for name, idx, t_b in (("ivf", ivf, t_ivf), ("hnsw", hnsw, t_hnsw)):
            if name == "ivf":
                idx.nprobe = 16
            _, I = idx.search(qv.astype(np.float32), args.k)
            out[f"faiss_{name}"] = {
                "build_s": t_b,
                "recall_cos": float(np.mean(
                    [len(set(I[i]) & set(cos_gt[i])) / args.k
                     for i in range(nq)])),
                "recall_jaccard": float(np.mean(
                    [len(set(I[i]) & set(jac_gt[i])) / args.k
                     for i in range(nq)])),
                "rss_mb_after_build": rss_mb()}
    except ImportError:
        out["faiss"] = "skipped: pip install faiss-cpu"

    # one JSON per method (per-config layout); --out keeps legacy
    # single-file mode
    if args.out:
        write_json(out_path, out)
    else:
        for method in ("minhash_lsh", "exact", "faiss_ivf", "faiss_hnsw"):
            if method in out:
                write_json(config_out("d1", f"{tag}_{method}")[0],
                           {"dataset": stem, "n_query": nq,
                            "method": method, **out[method]})
        if "faiss" in out:   # import-failure marker
            write_json(config_out("d1", f"{tag}_faiss")[0],
                       {"dataset": stem, "n_query": nq, **out})


if __name__ == "__main__":
    main()
