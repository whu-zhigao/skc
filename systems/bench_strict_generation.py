#!/usr/bin/env python3
"""Measure strict generation-tag overhead on real drift traces.

At each epoch, pages whose items cross theta are refreshed and receive a new
generation.  For every request and 32-query serving batch we count how many
neighbor sets span multiple page generations.  This quantifies the cost of
strict single-neighbor-generation assembly relative to DAI-2's dispersion mode.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mplignn.data import load_sessions  # noqa: E402
from systems.bench_retrieval import baseline_neighbor_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--drift", type=Path, required=True)
    ap.add_argument("--n-query", type=int, default=3000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--theta", type=float, default=0.10)
    ap.add_argument("--serving-batch", type=int, default=32)
    args = ap.parse_args()

    data = load_sessions(args.data)
    drift = np.load(args.drift)["drift"]
    queries = data.test[:min(args.n_query, len(data.test))]
    nbrs = baseline_neighbor_ids(queries, data.train, args.k)
    n_pages = (len(data.train) + args.page - 1) // args.page
    item_to_pages = defaultdict(set)
    for sid, session in enumerate(data.train):
        page = sid // args.page
        for item in set(session):
            item_to_pages[item].add(page)

    generation = np.zeros(n_pages, dtype=np.int32)
    epochs = []
    for interval, item_drift in enumerate(drift):
        moved = np.flatnonzero(item_drift > args.theta)
        dirty = set().union(*(item_to_pages[int(i)] for i in moved)) \
            if len(moved) else set()
        generation[list(dirty)] = interval + 1

        request_mixed, request_pages = [], []
        for nb in nbrs:
            pages = {sid // args.page for sid in nb}
            request_pages.append(len(pages))
            request_mixed.append(
                len({int(generation[p]) for p in pages}) > 1)

        batch_mixed, batch_pages = [], []
        for start in range(0, len(nbrs), args.serving_batch):
            group = nbrs[start:start + args.serving_batch]
            pages = {sid // args.page for nb in group for sid in nb}
            batch_pages.append(len(pages))
            batch_mixed.append(
                len({int(generation[p]) for p in pages}) > 1)

        epochs.append({
            "interval": interval,
            "dirty_pages": len(dirty),
            "dirty_page_fraction": len(dirty) / n_pages,
            "request_mixed_fraction": float(np.mean(request_mixed)),
            "mean_pages_per_request": float(np.mean(request_pages)),
            "serving_batch_mixed_fraction": float(np.mean(batch_mixed)),
            "mean_pages_per_serving_batch": float(np.mean(batch_pages)),
        })

    later = epochs[1:] or [{"request_mixed_fraction": 0.0,
                            "serving_batch_mixed_fraction": 0.0}]
    out = {
        "dataset": args.data.stem,
        "page": args.page,
        "theta": args.theta,
        "n_queries": len(nbrs),
        "serving_batch": args.serving_batch,
        "generation_tag_bytes": n_pages * 4,
        "epochs": epochs,
        "mean_request_mixed_fraction": float(np.mean(
            [e["request_mixed_fraction"] for e in later])),
        "mean_serving_batch_mixed_fraction": float(np.mean(
            [e["serving_batch_mixed_fraction"] for e in later])),
    }
    dest = Path("results/systems") / (
        f"strict_generation_{args.data.stem}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
