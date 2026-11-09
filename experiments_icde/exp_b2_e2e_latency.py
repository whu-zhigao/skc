#!/usr/bin/env python3
"""B2: end-to-end serving latency. Stages timed separately: profile
gather under cache states and placements (numpy CPU / pinned CPU /
GPU HBM), miss-path recompute, and model forward (via
common.run_forward -- if the signature does not match, stages (1)-(2)
still report and the script exits 0 with a warning)."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (config_out, load_data, load_model, neighbor_ids,
                    pool_profiles, run_forward, serving_prefixes, timeit,
                    write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--n-query", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="extra batch size swept for GPU saturation "
                         "(1/32/256 are always swept)")
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    ap.add_argument("--strict", action="store_true",
                    help="re-raise forward errors instead of recording "
                         "a skip (smoke tests should always use this)")
    args = ap.parse_args()

    stem = args.data.stem
    out_path, skip = config_out(
        "b2", f"{stem}_nq{args.n_query}_bsmax{args.batch_size}",
        args.out, args.force)
    if skip:
        print(f"[b2] exists, skipping: {out_path} (--force to rerun)")
        return

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(args.data)
    model = load_model(data, args.ckpt, device=dev)
    emb = model.embedding.weight.detach().cpu().numpy()

    nq = min(args.n_query, len(data.test))
    def _exactly_k(nb, qi):
        # run_forward pads to k_r too, but the flat-id gather below
        # REQUIRES len(ids) == bs*k for .view(bs, k, d); neighbor_ids
        # returns variable-length lists when a query has < k Jaccard
        # neighbors (the cause of the 2026-10-21 node skips).
        nb = np.asarray(nb, dtype=np.int64)
        if len(nb) == 0:
            return np.full(args.k, qi % len(data.train), dtype=np.int64)
        if len(nb) >= args.k:
            return nb[:args.k]
        reps = (args.k + len(nb) - 1) // len(nb)
        return np.tile(nb, reps)[:args.k]

    nbrs = [_exactly_k(nb, i) for i, nb in enumerate(
        neighbor_ids(serving_prefixes(data.test[:nq]), data.train, args.k))]
    print(f"[{stem}] {nq} queries, pooling corpus profiles...")
    P_cpu = pool_profiles(emb, data.train)
    P_t = torch.from_numpy(P_cpu)
    P_pin = P_t.pin_memory() if dev == "cuda" else P_t
    P_gpu = P_t.to(dev)
    d = P_cpu.shape[1]

    results = {"dataset": stem, "device": dev, "n_query": nq, "k": args.k,
               "batch_sizes": sorted({1, 32, 256, int(args.batch_size)})}
    fwd_failed = [False]
    for bs in sorted({1, 32, 256, int(args.batch_size)}):
        # must yield EXACTLY bs indices: arange(0,nq,stride)[:bs]
        # returns nq<bs entries and breaks view(bs,k,d) below
        # (the 2026-10-21 node failure: nq20/bs32 -> 12800 elems)
        qidx = (np.arange(bs) * max(1, nq // bs)) % nq
        ids = np.concatenate([nbrs[i] for i in qidx])
        results[f"gather_numpy_bs{bs}"] = timeit(
            lambda: P_cpu[ids], 20, 200)
        if dev == "cuda":
            ids_t = torch.from_numpy(ids).pin_memory()
            results[f"gather_pinned_bs{bs}"] = timeit(
                lambda: P_pin[ids_t], 20, 200)
            ids_g = ids_t.to(dev, non_blocking=True)
            results[f"gather_gpu_bs{bs}"] = timeit(
                lambda: P_gpu[ids_g], 20, 200)

        def miss_path():
            out = np.zeros((len(ids), d), dtype=np.float32)
            for j, sid in enumerate(ids):
                out[j] = emb[np.asarray(
                    sorted(set(data.train[sid])))].mean(axis=0)
            return out
        if bs <= 32:
            results[f"miss_recompute_bs{bs}"] = timeit(miss_path, 2, 10)

        try:
            seqs = [data.test[i][:-1] or data.test[i] for i in qidx]
            L = max(len(s) for s in seqs)
            sq = torch.zeros(bs, L, dtype=torch.long)
            for b, s in enumerate(seqs):
                sq[b, :len(s)] = torch.tensor(s)
            sq = sq.to(dev, non_blocking=True)
            prof = P_gpu[torch.from_numpy(ids).to(
                dev, non_blocking=True)].view(bs, args.k, d)
            results[f"forward_bs{bs}"] = timeit(
                lambda: run_forward(model, sq, prof), 5, 50)
        except RuntimeError as e:
            if args.strict:
                raise
            print(f"[WARN] forward skipped: {e}")
            results[f"forward_bs{bs}"] = {"skipped": str(e)}
            fwd_failed[0] = True

    results["forward_status"] = "failed" if fwd_failed[0] else "ok"
    write_json(out_path, results)


if __name__ == "__main__":
    main()
