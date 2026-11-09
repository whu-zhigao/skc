#!/usr/bin/env python3
"""Verify freeze/persist/load/delete lifecycle behavior."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mplignn.data import load_sessions  # noqa: E402
from systems.igms import ExactJaccard, IGMS, IGMSConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-query", type=int, default=100)
    args = ap.parse_args()
    data = load_sessions(args.data)
    queries = data.test[:min(args.n_query, len(data.test))]

    ig = IGMS(IGMSConfig(
        sig_len=64, bands=32, k_r=10, bucket_layout="sorted",
        keep_item_sets=True, compact_item_sets=True,
        discard_signatures=True,
    ))
    ig.insert_batch(data.train)
    ig.freeze()
    snapshot = Path("results/systems") / (
        f"posting_snapshot_{args.data.stem}.npz")
    t0 = time.perf_counter()
    ig.persist(snapshot)
    persist_s = time.perf_counter() - t0

    loaded = IGMS.load(snapshot)
    truth = ExactJaccard()
    for session in data.train:
        truth.insert(session)
    truth.finalize()
    recalls = []
    for query in queries:
        predicted = loaded.search(query, 10)
        actual = truth.topk(query, 10)
        recalls.append(len(set(predicted) & set(actual)) / 10)

    deleted = list(range(0, min(1000, len(data.train))))
    t0 = time.perf_counter()
    for sid in deleted:
        loaded.delete_session(sid)
    delete_s = time.perf_counter() - t0
    deleted_hits = sum(
        sid in loaded._candidates(query)
        for query in queries for sid in deleted
    )
    t0 = time.perf_counter()
    reclaimed = loaded.compact_postings()
    compact_s = time.perf_counter() - t0

    out = {
        "dataset": args.data.stem,
        "n_sessions": len(data.train),
        "n_queries": len(queries),
        "persist_s": persist_s,
        "snapshot_mb": snapshot.stat().st_size / 1e6,
        "load_recall_at_10": float(np.mean(recalls)),
        "delete_s": delete_s,
        "deleted_sessions": len(deleted),
        "deleted_candidate_hits_after_delete": deleted_hits,
        "posting_compaction_s": compact_s,
        "posting_bytes_reclaimed": reclaimed,
        "stable_process_portable_keys": True,
    }
    dest = Path("results/systems") / (
        f"posting_lifecycle_{args.data.stem}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    snapshot.unlink(missing_ok=True)
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
