#!/usr/bin/env python3
"""B7: staleness robustness of the neighbor cache. Builds pooled
neighbor profiles from two checkpoints (exp_a2's load_emb pattern) and
evaluates Recall@20 with the neighbor set mixed at stale ratios
0/25/50/75/100% (rows swapped to the old-generation table, seeded).
Optional --train: a light co-training loop (common.run_forward, next-
item CE) that refreshes the cached profiles every R steps for
R in {1,4,16,inf}; inf = never refresh after the initial (stale) build.
numpy-first; torch only for checkpoint loading and the --train loop."""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REAL, config_out, load_model, neighbor_ids,
                    pool_profiles, run_forward, write_json)
from exp_a2_drift_trace import load_emb


def top20_sets(Qn, Pn):
    idx = np.argpartition(-(Qn @ Pn.T), 20, axis=1)[:, :20]
    return [set(r) for r in idx]


def norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "--data", dest="dataset", type=Path,
                    required=True)
    ap.add_argument("--old-ckpt", type=Path, required=True)
    ap.add_argument("--new-ckpt", type=Path, default=None)
    ap.add_argument("--ratios", default="0,25,50,75,100")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--arm", choices=("cached", "nocache"), default="cached",
                    help="cached: neighbor profiles served from the "
                         "Tier-1 cache (refresh every R steps); nocache: "
                         "recompute profiles per batch from the live "
                         "table/raw sessions -- LIGHT-GRADE reference arm "
                         "(refresh cadence undefined, single run)")
    ap.add_argument("--ref-seeds", default="0,1,2,3,4",
                    help="seeds for the session-local reference row "
                         "(old-generation profiles vs exact top-20 on "
                         "the new-generation table, random nq-query test "
                         "subsets; mean+-std recorded in every JSON)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--cache-mb", type=float, default=5.0)
    ap.add_argument("--refresh-every", default="1,4,16,inf")
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="train-step batch size (GPU saturation; use 1 "
                         "for the original per-session steps)")
    ap.add_argument("--n-query", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun configs whose per-config JSON exists")
    args = ap.parse_args()

    new_ckpt = args.new_ckpt or (REAL / "abl_no_residual_diginetica.pt")
    missing = [str(p) for p in (args.dataset, args.old_ckpt, new_ckpt)
               if not Path(p).exists()]
    if missing:
        sys.exit("[b7] missing inputs:\n  " + "\n  ".join(missing)
                 + "\n       old/new checkpoints: pass --old-ckpt "
                   "(earlier-generation .pt with embedding.weight) and "
                   "--new-ckpt; generate snapshots with "
                   "exp_a2_drift_trace.py --ckpts DIR or inline "
                   "--snap-every, or reuse the B4 OLD_CKPT.")
    stem = args.dataset.stem
    ratios = [int(x) for x in args.ratios.split(",")]
    tag_mix = (f"{stem}_mix_r{{pct}}_nq{args.n_query}_s{args.seed}")
    if args.arm == "cached":
        tag_train = (f"{stem}_train_R{{r}}_e{args.epochs}"
                     f"_nq{args.n_query}_s{args.seed}")
    else:  # nocache arm: refresh cadence is undefined -> single run
        tag_train = (f"{stem}_train_{args.arm}_e{args.epochs}"
                     f"_nq{args.n_query}_s{args.seed}")
    pend_mix, mix_paths = [], {}
    for pct in ratios:
        p, skip = config_out("b7", tag_mix.format(pct=pct), None,
                             args.force)
        mix_paths[pct] = p
        if not skip:
            pend_mix.append(pct)
    pend_r, train_paths = [], {}
    if args.train:
        r_specs = ([r.strip() for r in args.refresh_every.split(",")]
                   if args.arm == "cached" else ["na"])
        for r in r_specs:
            p, skip = config_out("b7", tag_train.format(r=r), None,
                                 args.force)
            train_paths[r] = p
            if not skip:
                pend_r.append(r)
    if args.out:
        pend_mix = ratios
        pend_r = ([r.strip() for r in args.refresh_every.split(",")]
                  if args.arm == "cached" else ["na"])
    if not pend_mix and not pend_r:
        print("[b7] all configs exist, nothing to do (--force to rerun)")
        return
    from common import load_data
    rng = np.random.default_rng(args.seed)
    data = load_data(args.dataset)
    emb_new = load_emb(Path(new_ckpt))
    emb_old = load_emb(Path(args.old_ckpt))
    n, d = len(data.train), emb_new.shape[1]
    nq = min(args.n_query, len(data.test))
    queries = data.test[:nq]
    nbrs = neighbor_ids(queries, data.train, args.k)
    sigma = float(np.linalg.norm(emb_new - emb_old)
                  / np.linalg.norm(emb_old))
    print(f"[{stem}] {n} sessions d={d}, {nq} queries, "
          f"sigma_frobenius={sigma:.4f}")

    P_new = pool_profiles(emb_new, data.train)
    P_old = pool_profiles(emb_old, data.train)
    Qn = norm(pool_profiles(emb_new, queries))
    gt = top20_sets(Qn, norm(P_new))

    uniq = sorted({int(s) for nb in nbrs for s in nb})

    # ---- session-local reference row (--ref-seeds): recall of the
    # OLD-generation neighbor profiles against exact top-20 on the NEW
    # table, over seed-dependent random nq-query test subsets; the
    # seed-varied counterpart of the stale-100% mixing row.
    ref_seeds = [int(s) for s in args.ref_seeds.split(",") if s.strip()]
    ref_vals, ref_used = [], 0
    n_test = len(data.test)
    P_old_n, P_new_n = norm(P_old), norm(P_new)
    for s in ref_seeds:
        r2 = np.random.default_rng(10_000 + s)
        idx = r2.choice(n_test, size=min(nq, n_test), replace=False)
        sub = [data.test[int(i)] for i in idx]
        Qs = norm(pool_profiles(emb_new, sub))
        pred = top20_sets(Qs, P_old_n)
        gt_s = top20_sets(Qs, P_new_n)
        ref_vals.append(float(np.mean(
            [len(pred[i] & gt_s[i]) / 20 for i in range(len(sub))])))
        ref_used = len(sub)
    reference_row = {
        "definition": "R@20 of old-generation profiles vs exact top-20 "
                      "on the new-generation table; random nq-query "
                      "test subsets, one per seed",
        "seeds": ref_seeds, "n_query_per_seed": ref_used,
        "recall_at_20_mean": float(np.mean(ref_vals)),
        "recall_at_20_std": float(np.std(ref_vals)),
        "recall_at_20_per_seed": ref_vals}

    def hdr(extra):
        return {"dataset": stem, "n_query": nq, "k": args.k,
                "seed": args.seed, "arm": args.arm,
                "old_ckpt": str(args.old_ckpt),
                "new_ckpt": str(new_ckpt),
                "sigma_frobenius": sigma,
                "n_unique_neighbors": len(uniq),
                "reference_row": reference_row, **extra}

    md = ["| stale % | R@20 |", "|---|---|"]
    mix_rows = []
    for pct in ratios:
        if pct not in pend_mix:
            print(f"[b7] mix r{pct}%: exists, skipping {mix_paths[pct]}")
            continue
        f = pct / 100.0
        P = P_new.copy()
        if f > 0:
            stale = rng.choice(uniq, size=int(round(f * len(uniq))),
                               replace=False)
            P[stale] = P_old[stale]
        pred = top20_sets(Qn, norm(P))
        rec = float(np.mean([len(pred[i] & gt[i]) / 20
                             for i in range(nq)]))
        mix_rows.append({"stale_pct": pct, "recall_at_20": rec})
        print(f"  stale {pct:3d}%: R@20 {rec:.4f}")
        md.append(f"| {pct} | {rec:.4f} |")
        if not args.out:
            write_json(mix_paths[pct], hdr(
                {"config": "stale-mixing", "stale_pct": pct,
                 "recall_at_20": rec}))

    out = None
    if args.out:
        out = hdr({"mixing": mix_rows, "markdown_rows": md})

    if args.train and (args.out or pend_r):
        import torch
        torch.manual_seed(args.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        freq = Counter(s for nb in nbrs for s in nb)
        B = max(1, int(args.cache_mb * 1e6 / (d * 4)))
        cached = np.array([s for s, _ in freq.most_common(B)])
        print(f"[train] cache B={len(cached)} sessions "
              f"({args.cache_mb}MB), epochs={args.epochs}, "
              f"steps/epoch={args.steps_per_epoch}, "
              f"batch={args.batch_size}")
        train_out = {}
        try:
            sessions = [s if len(s) > 1 else s + s
                        for s in data.train]
            cached_sessions = [data.train[int(i)] for i in cached]
            # fixed per-session neighbor set (top-k of the initial,
            # stale cache) so the only varying factor is refresh cadence
            Pc0 = norm(pool_profiles(emb_old, cached_sessions))
            sess_emb = norm(pool_profiles(emb_old, sessions))
            nbr_fix = np.argpartition(
                -(sess_emb @ Pc0.T), args.k, axis=1)[:, :args.k]

            def to_dev(t):
                if dev == "cuda":
                    t = t.pin_memory()
                return t.to(dev, non_blocking=True)

            for Rspec in args.refresh_every.split(",") \
                    if args.arm == "cached" else ["na"]:
                Rspec = Rspec.strip()
                if Rspec not in pend_r and not args.out:
                    print(f"[b7] train R={Rspec}: exists, skipping "
                          f"{train_paths[Rspec]}")
                    continue
                R = (int(Rspec) if Rspec not in ("inf", "na")
                     else float("inf"))
                model = load_model(data, Path(new_ckpt), device=dev)
                model.train()
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                P_cache = pool_profiles(emb_old, cached_sessions)
                n_refresh, t_refresh, losses = 0, 0.0, []
                r_rows = []
                for ep in range(1, args.epochs + 1):
                    for step in range(1, args.steps_per_epoch + 1):
                        if args.arm == "cached" and step % R == 0:
                            t0 = time.perf_counter()
                            emb_cur = (model.embedding.weight.detach()
                                       .cpu().numpy())
                            P_cache = pool_profiles(
                                emb_cur, cached_sessions)
                            t_refresh += time.perf_counter() - t0
                            n_refresh += 1
                        js = rng.integers(
                            0, len(sessions),
                            size=min(args.batch_size, len(sessions)))
                        batch = [sessions[int(j)] for j in js]
                        L = max(len(s) for s in batch)
                        bs = len(batch)
                        seq = torch.zeros(bs, L - 1, dtype=torch.long)
                        tgt = torch.zeros(bs, dtype=torch.long)
                        for b, s in enumerate(batch):
                            seq[b, :len(s) - 1] = torch.tensor(s[:-1])
                            tgt[b] = s[-1]
                        if args.arm == "cached":
                            prof = torch.from_numpy(P_cache[nbr_fix[js]])
                        else:
                            # nocache arm: recompute this batch's
                            # neighbor profiles from the LIVE table
                            # (raw sessions) -- the no-Tier-1-cache
                            # reference path. LIGHT-GRADE.
                            emb_cur = (model.embedding.weight.detach()
                                       .cpu().numpy())
                            flat = nbr_fix[js].reshape(-1)
                            prof = torch.from_numpy(pool_profiles(
                                emb_cur,
                                [cached_sessions[int(i)]
                                 for i in flat]).reshape(bs, -1, d))
                        out_f = run_forward(model, to_dev(seq),
                                            to_dev(prof))
                        lg = (out_f[0] if isinstance(out_f, tuple)
                              else out_f).reshape(bs, -1).float()
                        loss = torch.nn.functional.cross_entropy(
                            lg[:, :data.n_items], to_dev(tgt))
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        losses.append(float(loss))
                    emb_cur = model.embedding.weight.detach().cpu().numpy()
                    P_all = pool_profiles(emb_cur, data.train)
                    P_eval = P_all.copy()
                    if args.arm == "cached":
                        P_eval[cached] = P_cache
                    pred = top20_sets(norm(pool_profiles(
                        emb_cur, queries)), norm(P_eval))
                    rec = float(np.mean(
                        [len(pred[i] & gt[i]) / 20 for i in range(nq)]))
                    r_rows.append({
                        "epoch": ep, "recall_at_20": rec,
                        "n_refreshes": n_refresh,
                        "refresh_s": round(t_refresh, 4),
                        "mean_loss": float(np.mean(losses[-50:]))})
                    print(f"  R={Rspec} ep{ep}: R@20 {rec:.4f} "
                          f"refreshes={n_refresh} "
                          f"({t_refresh:.2f}s)")
                train_cfg = {
                    "cache_sessions": int(len(cached)),
                    "cache_mb": args.cache_mb,
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size": args.batch_size, "lr": 1e-3,
                    "arm": args.arm,
                    "grade": "LIGHT" if args.arm == "nocache"
                             else "normal",
                    "note": ("neighbor sets fixed at the initial stale "
                             "cache; only the refresh cadence R varies"
                             if args.arm == "cached" else
                             "NOCACHE arm (LIGHT-GRADE): neighbor "
                             "profiles recomputed from the live table "
                             "per batch; refresh cadence undefined. "
                             "Recall@20 ground truth = exact top-20 on "
                             "the CURRENT full-precision table each "
                             "epoch.")}
                if not args.out:
                    write_json(train_paths[Rspec], hdr(
                        {"config": "train-refresh", "refresh_every":
                         Rspec, "epochs": r_rows,
                         "train_config": train_cfg}))
                else:
                    train_out[Rspec] = {"epochs": r_rows}
        except (RuntimeError, TypeError) as e:
            print(f"[WARN] train loop skipped: {e}")
            train_out = f"skipped: {e}"

        if args.out:
            out["train"] = train_out
            if isinstance(train_out, dict):
                out["train_config"] = {
                    "cache_mb": args.cache_mb, "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size": args.batch_size, "lr": 1e-3,
                    "note": "neighbor sets fixed at the initial stale "
                            "cache; only the refresh cadence R varies."}

    if args.out:
        write_json(args.out, out)


if __name__ == "__main__":
    main()
