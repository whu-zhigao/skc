#!/usr/bin/env python3
"""B6: precision spectrum for Tier-1 pooled profiles. Configs: int8-all,
int4-all, int2-all, utility-mixed (top 25% by NFA utility -> int8, mid
50% -> int4, tail 25% -> int2) and elbow-mixed (data-driven knee). For
each: analytic bytes, mean/max cosine error of dequantized profiles vs
fp32, and downstream Recall@20 of cosine retrieval on the dequantized
cache vs exact fp32 ground truth. Drift-rescale study: perturb the
embedding table (relative sigma = --drift) and compare stale codes+
scales, rescale-only (keep int codes, refresh per-row scales from the
drifted rows) and full re-quantization. numpy-only; torch appears only
inside exp_a2's load_emb for the checkpoint."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REAL, config_out, dequantize_rows,
                    elbow_mixed_partition, mixed_precision_partition,
                    mixed_rows_bytes, neighbor_ids, serving_prefixes, pool_profiles,
                    quantize_rows, quantized_rows_bytes, write_json)
from exp_a2_drift_trace import load_emb


def quantize_mixed(P, bits):
    """Quantize rows at their per-row widths; returns codes (int8),
    scales and total analytic bytes."""
    Xq = np.empty_like(P)
    codes = np.empty(P.shape, dtype=np.int8)
    scales = np.empty((P.shape[0], 1), dtype=np.float32)
    total = 0
    for b in (8, 4, 2):
        idx = np.flatnonzero(bits == b)
        if idx.size == 0:
            continue
        c, s = quantize_rows(P[idx], b)
        codes[idx], scales[idx] = c, s
        Xq[idx] = dequantize_rows(c, s, b)
        total += quantized_rows_bytes(idx.size, P.shape[1], b)
    return codes, scales, Xq, total


def dequantize_mixed(codes, scales, bits):
    """Dequantize with each group's own qmax (per-row bit widths)."""
    Xr = np.empty(codes.shape, dtype=np.float32)
    for b in (8, 4, 2):
        idx = np.flatnonzero(bits == b)
        if idx.size == 0:
            continue
        Xr[idx] = dequantize_rows(codes[idx], scales[idx], b)
    return Xr


def rel_l2(Xr, X):
    """Per-row relative L2 error ||Xr-X||/||X|| (scale-sensitive)."""
    return (np.linalg.norm(Xr - X, axis=1)
            / (np.linalg.norm(X, axis=1) + 1e-12))


def cos_err(A, B):
    an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.sum(an * bn, axis=1)


def top20(Qn, Pn):
    return np.argpartition(-Qn @ Pn.T, 20, axis=1)[:, :20]


def e2e_eval(model, data, queries, nbrs, prof_matrix, k, n_items):
    """End-to-end quality of the REAL model (common.run_forward) with
    neighbor profiles served from `prof_matrix` (dequantized cache rows
    or fp32 reference): HR@20 / MRR@10 on next-item prediction over the
    test queries. LIGHT-GRADE (small n_query, CPU-friendly batches)."""
    import torch
    from common import ranking_metrics, run_forward
    use = [i for i, q in enumerate(queries) if len(q) >= 2]
    hrs, mrrs = [], []
    bs = 32
    for i0 in range(0, len(use), bs):
        idx = use[i0:i0 + bs]
        qs = [queries[i] for i in idx]
        L = max(len(s) for s in qs)
        seq = torch.zeros(len(qs), L - 1, dtype=torch.long)
        tgt = torch.zeros(len(qs), dtype=torch.long)
        for b, s in enumerate(qs):
            seq[b, :len(s) - 1] = torch.tensor(s[:-1])
            tgt[b] = s[-1]
        prof = torch.from_numpy(prof_matrix[
            np.asarray([nbrs[i] for i in idx], dtype=np.int64)]).float()
        logits = run_forward(model, seq, prof).detach().cpu().numpy()
        m = ranking_metrics(logits[:, :n_items], tgt.numpy(), ks=(20,))
        hrs.append(m["HR@20"])
        mrrs.append(m["MRR@10"])
    return {"n_query": len(use),
            "hr_at_20": float(np.mean(hrs)),
            "mrr_at_10": float(np.mean(mrrs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "--data", dest="dataset", type=Path,
                    required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--configs", default="int8-all,int4-all,int2-all,"
                                         "utility-mixed,elbow-mixed")
    ap.add_argument("--drift", type=float, default=1e-2,
                    help="relative perturbation sigma on embeddings")
    ap.add_argument("--e2e", action="store_true",
                    help="additionally push the DEQUANTIZED profiles "
                         "through the real model (common.run_forward) "
                         "and report HR@20/MRR@10 next-item metrics vs "
                         "the fp32-profiles reference (LIGHT-GRADE; "
                         "requires the per-config file to be absent or "
                         "--force, since e2e is appended to it)")
    ap.add_argument("--n-query", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun configs whose per-config JSON exists")
    args = ap.parse_args()

    ckpt = args.ckpt or (REAL / "abl_no_residual_diginetica.pt")
    if not args.dataset.exists():
        sys.exit(f"[b6] dataset not found: {args.dataset}\n"
                 f"       pass --dataset <repo>/code/data/diginetica.pkl")
    if not Path(ckpt).exists():
        sys.exit(f"[b6] checkpoint not found: {ckpt}\n"
                 f"       pass --ckpt <model.pt> (embedding.weight state "
                 f"dict, as produced by the training runs; a bare tensor "
                 f"is also accepted)")
    stem = args.dataset.stem
    cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
    tag_base = f"{stem}_{{cfg}}_nq{args.n_query}_dr{args.drift:g}_s{args.seed}"
    pending, paths = [], {}
    for cfg in cfgs:
        p, skip = config_out("b6", tag_base.format(cfg=cfg), None,
                             args.force)
        paths[cfg] = p
        if not skip:
            pending.append(cfg)
    if args.out:
        pending = cfgs           # legacy mode: run everything
    if not pending:
        print("[b6] all configs exist, nothing to do (--force to rerun)")
        return
    from common import load_data
    rng = np.random.default_rng(args.seed)
    data = load_data(args.dataset)
    emb = load_emb(Path(ckpt))
    n, d = len(data.train), emb.shape[1]
    nq = min(args.n_query, len(data.test))
    queries = serving_prefixes(data.test[:nq])
    print(f"[{stem}] {n} profiles d={d}, {nq} queries, "
          f"drift sigma_rel={args.drift}")

    # NFA utility proxy = full-trace neighbor frequency (B1 oracle).
    nbrs = neighbor_ids(queries, data.train, args.k)
    util = np.zeros(n)
    for sid, c in Counter(s for nb in nbrs for s in nb).items():
        util[sid] = c

    P = pool_profiles(emb, data.train)
    Q = pool_profiles(emb, queries)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    gt = top20(Qn, Pn)
    gt_sets = [set(g) for g in gt]

    P_drift = (P + args.drift * float(P.std())
               * rng.standard_normal(P.shape)).astype(np.float32)

    e2e_model, e2e_fp32 = None, {}
    if args.e2e:
        from common import load_model
        e2e_model = load_model(data, Path(ckpt), device="cpu")
        e2e_model.eval()

    rows, md = [], ["| config | mean bits | MB | cosErr mean | cosErr "
                    "max | R@20 | drift relL2: stale/rescale/requant |",
                    "|---|---|---|---|---|---|---|"]
    for cfg in cfgs:
        if cfg not in pending:
            print(f"[b6] {cfg}: exists, skipping {paths[cfg]}")
            continue
        if cfg.endswith("-all"):
            try:
                bits_n = int(cfg[3:-4])   # "int8-all" -> 8
            except ValueError:
                sys.exit(f"[b6] bad config {cfg!r} (expected int8-all / "
                         f"int4-all / int2-all)")
            bits = np.full(n, bits_n, dtype=np.int8)
        elif cfg == "utility-mixed":
            bits = mixed_precision_partition(util)
        elif cfg == "elbow-mixed":
            bits = elbow_mixed_partition(util)
        else:
            sys.exit(f"[b6] unknown config {cfg!r}")
        codes, scales, Xq, nbytes = quantize_mixed(P, bits)
        Xqn = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-12)
        pred = top20(Qn, Xqn)
        rec = float(np.mean([len(set(pred[i]) & gt_sets[i]) / 20
                             for i in range(nq)]))
        ce = cos_err(P, Xq)

        # ---- drift-rescale arms (per config, vs the drifted rows)
        # cosine error is direction-only; rescale-only refreshes per-row
        # magnitudes, so the discriminative metric is relative L2.
        arm = {}
        for name in ("stale", "rescale", "requant"):
            if name == "stale":
                Xr = dequantize_mixed(codes, scales, bits)
            elif name == "rescale":
                s2 = np.empty_like(scales)
                for b in (8, 4, 2):
                    idx = np.flatnonzero(bits == b)
                    if idx.size == 0:
                        continue
                    qmax = float(2 ** (b - 1) - 1)
                    s2[idx] = np.maximum(
                        np.abs(P_drift[idx]).max(axis=1, keepdims=True),
                        1e-12) / qmax
                Xr = dequantize_mixed(codes, s2, bits)
            else:
                _, _, Xr, _ = quantize_mixed(P_drift, bits)
            e = cos_err(P_drift, Xr)
            l2 = rel_l2(Xr, P_drift)
            arm[name] = {"cos_err_mean": float(e.mean()),
                         "cos_err_max": float(e.max()),
                         "rel_l2_mean": float(l2.mean()),
                         "rel_l2_max": float(l2.max())}
        row = {"config": cfg, "mean_bits": float(np.mean(bits)),
               "bytes": nbytes, "mb": nbytes / 1e6,
               "cos_err_mean": float(ce.mean()),
               "cos_err_max": float(ce.max()), "recall_at_20": rec,
               "drift_arms": arm}
        e2e_block = None
        if args.e2e:
            if "hr_at_20" not in e2e_fp32:     # fp32 reference once
                e2e_fp32.update(e2e_eval(
                    e2e_model, data, queries, nbrs, P, args.k,
                    data.n_items))
            qm = e2e_eval(e2e_model, data, queries, nbrs, Xq, args.k,
                          data.n_items)
            e2e_block = {**qm,
                         "hr_at_20_fp32_ref": e2e_fp32["hr_at_20"],
                         "mrr_at_10_fp32_ref": e2e_fp32["mrr_at_10"],
                         "ckpt": str(ckpt), "grade": "LIGHT",
                         "note": "HR@20/MRR@10 of the real model with "
                                 "neighbor profiles served from the "
                                 "dequantized cache rows; fp32_ref = "
                                 "same forward with fp32 profiles"}
        rows.append(row)
        print(f"  {cfg:>13s}: {row['mean_bits']:.2f} bits "
              f"{row['mb']:.1f}MB cosErr {row['cos_err_mean']:.4f} "
              f"R@20 {rec:.3f} | drift relL2 s/r/q "
              f"{arm['stale']['rel_l2_mean']:.4f}/"
              f"{arm['rescale']['rel_l2_mean']:.4f}/"
              f"{arm['requant']['rel_l2_mean']:.4f}")
        md.append(f"| {cfg} | {row['mean_bits']:.2f} | {row['mb']:.1f} "
                  f"| {row['cos_err_mean']:.4f} | "
                  f"{row['cos_err_max']:.4f} | {rec:.3f} | "
                  f"{arm['stale']['rel_l2_mean']:.4f}/"
                  f"{arm['rescale']['rel_l2_mean']:.4f}/"
                  f"{arm['requant']['rel_l2_mean']:.4f} |")
        if not args.out:
            payload = {
                "dataset": stem, "n_profiles": n, "d": d,
                "n_query": nq, "ckpt": str(ckpt),
                "drift_rel_sigma": args.drift, "seed": args.seed,
                "config": row,
                "note": "Per-row symmetric quantization "
                        "(common.quantize_rows): scale = max|row|/qmax "
                        "stored fp32; bytes are analytic (codes + "
                        "scale). Recall@20 = overlap of top-20 cosine "
                        "retrieval on the dequantized cache with exact "
                        "fp32 top-20. Drift arms: stale (old "
                        "codes+scales), rescale (old codes, scales "
                        "refreshed), requant (re-quantize); rel_l2 is "
                        "the scale-sensitive metric."}
            if e2e_block is not None:
                payload["e2e"] = e2e_block
            write_json(paths[cfg], payload)
    print("\n".join(md))

    if not args.out:
        return
    write_json(args.out, {
        "dataset": stem, "n_profiles": n, "d": d, "n_query": nq,
        "ckpt": str(ckpt), "drift_rel_sigma": args.drift,
        "configs": rows, "markdown_rows": md,
        "note": "Per-row symmetric quantization (common.quantize_rows): "
                "scale = max|row|/qmax stored fp32; bytes are analytic "
                "(codes + scale). Recall@20 = overlap of top-20 cosine "
                "retrieval on the dequantized cache with exact fp32 "
                "top-20. Drift: X' = X + drift*std(X)*N(0,1); arms are "
                "stale (old codes+scales), rescale (old codes, scales "
                "refreshed from X') and requant (full re-quantize of "
                "X'). cos_err is direction-only, so stale and rescale "
                "coincide on it by construction (rescale multiplies "
                "each row by a scalar); rel_l2 = ||Xr-X'||/||X'|| is "
                "the scale-sensitive metric that separates the arms."})


if __name__ == "__main__":
    main()
