#!/usr/bin/env python3
"""B3: DAI skip-rate and savings replayed over the REAL drift trace
from A2 (drift_trace_<stem>.npz). Reports dirty-page fractions for a
theta sweep, measures true per-page recompute cost on this node,
projects per-epoch savings; --live additionally times full-invalidate
vs DAI maintenance inside inline training."""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (config_out, ensure_results, load_data, load_model,
                    write_json)


def page_cost_ms(data, emb, G=128, iters=50):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for sid in range(0, min(G * 4, len(data.train)), 4):
            _ = emb[np.asarray(sorted(set(data.train[sid])))].mean(axis=0)
        ts.append(time.perf_counter() - t0)   # time to pool G sessions
    return 1000.0 * float(np.median(ts))      # ms per G-session page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--trace", type=Path, default=None)
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="model checkpoint; default is the diginetica "
                         "one in common.load_model — pass the "
                         "same-dataset ckpt for gowalla/yoochoose")
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()
    stem = args.data.stem
    out_path, skip = config_out(
        "b3", f"{stem}_p{args.page}" + ("_live" if args.live else ""),
        args.out, args.force)
    if skip:
        print(f"[b3] exists, skipping: {out_path} (--force to rerun)")
        return
    data = load_data(args.data)
    trace = args.trace or (ensure_results() / f"drift_trace_{stem}.npz")
    drift = np.load(trace)["drift"]
    T, n_items = drift.shape
    assert n_items == data.n_items, "trace/model item count mismatch"

    item2pages = defaultdict(set)
    for sid, s in enumerate(data.train):
        pg = sid // args.page
        for it in set(s):
            item2pages[it].add(pg)
    n_pages = len(data.train) // args.page + 1

    # theta sweep: quantile-derived thetas carry a q-label (the paper's
    # theta must be re-anchored to a trace quantile -- see
    # BACKFILL_MAP); fixed literals keep the "abs" label.
    theta_specs = [(f"q{q}", float(np.percentile(drift, q)))
                   for q in (10, 25, 50, 75, 90, 95, 99)]
    theta_specs += [("abs", v) for v in (0.001, 0.01, 0.1, 0.5)]
    seen_th = set()
    rows = []
    for label, th in theta_specs:
        key = round(th, 12)
        if key in seen_th:            # dedupe identical thetas
            continue
        seen_th.add(key)
        skips = []
        for t in range(T):
            moved = np.flatnonzero(drift[t] > th)
            dirty = set().union(*(item2pages[i] for i in moved)) \
                if len(moved) else set()
            skips.append(1.0 - len(dirty) / n_pages)
        rows.append({"theta": th, "theta_label": label,
                     "mean_skip_rate": float(np.mean(skips))})
        print(f"  theta={th:.4f} [{label}]: skip {np.mean(skips):.3f}")

    model = load_model(data, args.ckpt)
    emb = model.embedding.weight.detach().numpy()
    cost = page_cost_ms(data, emb, G=args.page)
    out = {"dataset": stem, "page": args.page, "trace": str(trace),
           "n_pages": n_pages, "n_intervals": T,
           "measured_page_recompute_ms": cost,
           "theta_sweep": rows,
           "projected_epoch_saving_s": [
               {"theta": r["theta"], "theta_label": r["theta_label"],
                "saving_s": r["mean_skip_rate"] * n_pages * cost / 1000.0}
               for r in rows],
           "live_mode": bool(args.live)}

    if args.live:
        from exp_a2_drift_trace import inline_train
        th = float(np.percentile(drift, 75))
        npages_live = 64
        pages = {p: list(range(p * args.page, (p + 1) * args.page))
                 for p in range(npages_live)}
        item2pages_live = defaultdict(set)
        for sid in range(npages_live * args.page):
            for it in set(data.train[sid]):
                item2pages_live[it].add(sid // args.page)

        def recompute(emb_now, sess_ids):
            for sid in sess_ids:
                _ = emb_now[np.asarray(
                    sorted(set(data.train[sid])))].mean(axis=0)

        t_full = t_dai = 0.0
        emb_now = emb
        for new, old in inline_train(data, steps=600, snap_every=50)[1:]:
            moved = np.flatnonzero(
                np.linalg.norm(new - old, axis=1) > th)
            dirty = {p for it in moved
                     for p in item2pages_live.get(it, ())}
            emb_now = new
            t0 = time.perf_counter()
            for v in pages.values():
                recompute(emb_now, v)
            t_full += time.perf_counter() - t0
            t0 = time.perf_counter()
            for p, v in pages.items():
                if p in dirty:
                    recompute(emb_now, v)
            t_dai += time.perf_counter() - t0
        out["live"] = {"theta_q75": th, "pages": npages_live,
                       "full_invalidate_s": t_full, "dai_s": t_dai,
                       "speedup": t_full / max(t_dai, 1e-9)}
    write_json(out_path, out)


if __name__ == "__main__":
    main()
