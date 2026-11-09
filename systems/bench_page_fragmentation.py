#!/usr/bin/env python3
"""Measure session admission versus page-level residency fragmentation."""
from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_icde.common import (serving_prefixes,  # noqa: E402
                                     neighbor_ids)
from mplignn.data import load_sessions  # noqa: E402


def lru_final_set(nbrs, capacity: int) -> set[int]:
    cache = OrderedDict()
    for nb in nbrs:
        for sid in nb:
            if sid in cache:
                cache.move_to_end(sid)
            else:
                cache[sid] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    return set(cache)


def page_summary(selected, sessions, page_size: int):
    pages = {sid // page_size for sid in selected}
    selected_in_pages = sum(
        1 for sid, _ in enumerate(sessions) if sid // page_size in pages
    )
    per_page = Counter(sid // page_size for sid in selected)
    touched_values = list(per_page.values())
    ledger_entries = sum(
        len({item for session in sessions[page * page_size:
                                           (page + 1) * page_size]
             for item in session})
        for page in pages
    )
    return {
        "page_size": page_size,
        "selected_sessions": len(selected),
        "touched_pages": len(pages),
        "total_pages": (len(sessions) + page_size - 1) // page_size,
        "touched_page_fraction": len(pages) /
        ((len(sessions) + page_size - 1) // page_size),
        "mean_selected_per_touched_page": float(np.mean(touched_values)),
        "max_selected_per_touched_page": int(max(touched_values, default=0)),
        "sessions_resident_if_full_pages": selected_in_pages,
        "session_payload_mb": len(selected) * 68 / 1e6,
        "full_page_payload_mb": selected_in_pages * 68 / 1e6,
        "payload_amplification": (selected_in_pages / len(selected)
                                  if selected else 0.0),
        "page_ledger_entries": ledger_entries,
        "page_ledger_mb": ledger_entries * 8 / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n-query", type=int, default=3000)
    parser.add_argument("--budgets-mb", default="1,2,5,10")
    parser.add_argument("--page-sizes", default="64,128,256,512")
    args = parser.parse_args()

    data = load_sessions(args.data)
    nq = min(args.n_query, len(data.test))
    queries = serving_prefixes(data.test[:nq])
    nbrs = neighbor_ids(queries, data.train, 10)
    freq = Counter(sid for nb in nbrs for sid in nb)
    active = set(freq)
    rows = []

    for budget_mb in [float(x) for x in args.budgets_mb.split(",")]:
        capacity = int(budget_mb * 1e6 / (64 * 4))
        nfa = {sid for sid, _ in freq.most_common(capacity)}
        lru = lru_final_set(nbrs, capacity)
        rng = np.random.default_rng(7)
        random_ids = set(rng.choice(sorted(active),
                                    size=min(capacity, len(active)),
                                    replace=False).tolist())
        for mode, selected in (("NFA-profiled", nfa),
                               ("LRU-online-final", lru),
                               ("random-active", random_ids),
                               ("active-ceiling", active)):
            for page_size in [int(x) for x in args.page_sizes.split(",")]:
                row = {
                    "budget_mb": budget_mb,
                    "mode": mode,
                    **page_summary(selected, data.train, page_size),
                }
                rows.append(row)

    result = {
        "dataset": args.data.stem,
        "n_queries": nq,
        "k_neighbors": 10,
        "active_sessions": len(active),
        "protocol": ("prefix-only exact-Jaccard neighbors; session payload "
                     "is 64-B int8 code + 4-B scale"),
        "rows": rows,
    }
    dest = Path("results/systems") / f"page_fragmentation_{args.data.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
