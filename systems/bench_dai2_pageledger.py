#!/usr/bin/env python3
"""Exact page-local DAI-2 ledger replay.

Unlike the earlier diagnostic in experiments_icde/exp_b9, this benchmark does
not reset a global per-item accumulator.  Every page owns its own ledger;
refreshing one page resets only that page's entries.  The result is the
refresh volume and ledger footprint implied by the contract used in the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mplignn.data import load_sessions  # noqa: E402


def build_pages(sessions: Sequence[Sequence[int]], page_size: int):
    pages: List[np.ndarray] = []
    for start in range(0, len(sessions), page_size):
        chunk = sessions[start:start + page_size]
        items = sorted({int(i) for session in chunk for i in session})
        pages.append(np.asarray(items, dtype=np.int32))
    return pages


def replay(page_items: List[np.ndarray], drift: np.ndarray,
           theta: float, mode: str = "page_local") -> Dict:
    ledgers = [np.zeros(items.size, dtype=np.float64)
               for items in page_items]
    item_ledger = np.zeros(drift.shape[1], dtype=np.float32)
    per_interval = []
    total_refreshes = 0
    unique_dirty = set()

    for interval, row in enumerate(drift):
        dirty = []
        for pid, items in enumerate(page_items):
            if not items.size:
                continue
            if mode == "page_local":
                ledgers[pid] += row[items]
                crossed = bool(ledgers[pid].max() > theta)
            else:
                # In the per_epoch arm, only the current update matters.  The
                # global_item arm fills its ledgers below.
                crossed = bool(items.size and row[items].max() > theta)
            if crossed:
                dirty.append(pid)

        if mode == "global_item":
            item_ledger += row
            moved = np.flatnonzero(item_ledger > theta)
            dirty = sorted({pid for pid, items in enumerate(page_items)
                            if items.size and
                            np.intersect1d(items, moved, assume_unique=False).size})
            item_ledger[moved] = 0.0
        else:
            for pid in dirty:
                ledgers[pid].fill(0.0)

        total_refreshes += len(dirty)
        unique_dirty.update(dirty)
        per_interval.append({
            "interval": interval,
            "dirty_pages": len(dirty),
            "dirty_fraction": len(dirty) / max(1, len(page_items)),
        })

    return {
        "mode": mode,
        "theta": theta,
        "intervals": int(drift.shape[0]),
        "pages": len(page_items),
        "total_page_refreshes": total_refreshes,
        "full_invalidation_refreshes": len(page_items) * int(drift.shape[0]),
        "mean_refreshes_per_interval": total_refreshes / max(1, len(drift)),
        "unique_dirty_pages": len(unique_dirty),
        "per_interval": per_interval,
        "ledger_accounted_mb": (
            sum(x.nbytes for x in page_items) +
            sum(x.nbytes for x in ledgers)) / 1e6,
        "membership_mb": sum(x.nbytes for x in page_items) / 1e6,
        "ledger_mb": sum(x.nbytes for x in ledgers) / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--drift", type=Path, required=True)
    parser.add_argument("--thetas", default="0.05,0.10,0.15")
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = load_sessions(args.data)
    drift = np.load(args.drift)["drift"].astype(np.float64)
    page_items = build_pages(data.train, args.page_size)
    result = {
        "dataset": args.data.stem,
        "page_size": args.page_size,
        "n_sessions": len(data.train),
        "n_intervals": int(drift.shape[0]),
        "n_items": int(drift.shape[1]),
        "page_membership_entries": int(sum(x.size for x in page_items)),
        "thetas": {},
    }
    for theta in [float(x) for x in args.thetas.split(",")]:
        t0 = time.perf_counter()
        local = replay(page_items, drift, theta, mode="page_local")
        local["replay_seconds"] = time.perf_counter() - t0
        per_epoch = replay(page_items, drift, theta, mode="per_epoch")
        legacy = replay(page_items, drift, theta, mode="global_item")
        result["thetas"][f"{theta:.3f}"] = {
            "page_local": local,
            "per_epoch": per_epoch,
            "legacy_global_item_diagnostic": legacy,
        }
        print(f"{args.data.stem} theta={theta:.2f}: "
              f"{local['total_page_refreshes']} page refreshes, "
              f"{local['ledger_accounted_mb']:.2f} MB ledger")

    out = args.out or (ROOT / "results" / "systems" /
                       f"dai2_pageledger_{args.data.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
