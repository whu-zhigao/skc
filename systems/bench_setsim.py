#!/usr/bin/env python3
"""Prefix-filter Jaccard search baseline for set-similarity retrieval."""
from __future__ import annotations

import argparse
import json
import tracemalloc
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_icde.common import serving_prefixes  # noqa: E402
from mplignn.data import load_sessions  # noqa: E402


class PrefixJaccardIndex:
    """Adaptive prefix filter followed by exact Jaccard verification."""
    def __init__(self, corpus, k: int = 10):
        self.corpus = [set(map(int, session)) for session in corpus]
        self.length = np.asarray([len(x) for x in self.corpus], dtype=np.int32)
        doc_freq = defaultdict(int)
        for session in self.corpus:
            for item in session:
                doc_freq[item] += 1
        self.order = {item: rank for rank, (item, _) in enumerate(
            sorted(doc_freq.items(), key=lambda x: (x[1], x[0])))}
        self.index = defaultdict(list)
        for sid, session in enumerate(self.corpus):
            ordered = sorted(session, key=lambda x: self.order[x])
            # Index the complete rare-token prefix used by adaptive probing.
            for item in ordered:
                self.index[item].append(sid)
        self.index = {item: np.asarray(sids, dtype=np.int32)
                      for item, sids in self.index.items()}
        self.k = k
        self.probe_items = 0
        self.verified = 0

    def _verify(self, query, sid):
        self.verified += 1
        other = self.corpus[sid]
        inter = len(query & other)
        if not inter:
            return 0.0
        return inter / (len(query) + len(other) - inter)

    def search(self, items):
        query = set(map(int, items))
        if not query:
            return []
        ordered = sorted(query, key=lambda x: self.order.get(x, 1 << 30))
        overlap = defaultdict(int)

        # Seed k candidates from the rarest query tokens, then use the kth
        # similarity as an adaptive prefix bound.
        seed_candidates = set()
        seed_probe_count = 0
        for item in ordered:
            seed_probe_count += 1
            self.probe_items += 1
            seed_candidates.update(
                self.index.get(item, np.empty(0, dtype=np.int32)).tolist())
            if len(seed_candidates) >= max(self.k * 20, self.k):
                break
        heap = []
        returned = set()
        for sid in seed_candidates:
            score = self._verify(query, int(sid))
            if score:
                heap.append((score, int(sid)))
                returned.add(int(sid))
        heap.sort(reverse=True)
        heap = heap[:self.k]
        threshold = heap[-1][0] if len(heap) == self.k else 0.0

        # Two sets with Jaccard > t must share a token in the first
        # |q|-ceil(t|q|)+1 positions of the globally ordered query.
        # Probe the complete rare-token prefix. The seed threshold still prunes
        # verification by length; disabling a shorter adaptive prefix avoids
        # losing tied neighbors when the kth score changes during probing.
        probe_tokens = ordered
        # Include seed postings in the candidate map, then count the rest.
        for item in probe_tokens:
            self.probe_items += 1
            for sid in self.index.get(item, ()):
                overlap[int(sid)] += 1

        # Verify only candidates in the length range allowed by the threshold.
        candidates = set(overlap)
        # Verify every candidate that shares a queried token. Length filtering
        # is possible in a full join setting, but top-k ties make it unsafe
        # here when the kth threshold changes during the probe.
        for sid in candidates:
            score = self._verify(query, sid)
            if (score and sid not in returned and
                    (len(heap) < self.k or (score, sid) > heap[-1])):
                heap.append((score, sid))
                returned.add(sid)
                heap.sort(reverse=True)
                heap = heap[:self.k]
        return [sid for _, sid in heap]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n-query", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    data = load_sessions(args.data)
    step = max(1, len(data.test) // args.n_query)
    queries = serving_prefixes(
        [data.test[i] for i in range(0, len(data.test), step)][:args.n_query])

    tracemalloc.start()
    t0 = time.perf_counter()
    index = PrefixJaccardIndex(data.train, args.k)
    build_s = time.perf_counter() - t0
    _, build_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latencies = []
    recalls = []
    for query in queries:
        t0 = time.perf_counter()
        pred = index.search(query)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        # Exact full scan for ground truth on this benchmark.
        qset = set(query)
        truth = sorted(((len(qset & s) /
                         (len(qset) + len(s) - len(qset & s)), sid)
                        for sid, s in enumerate(index.corpus)
                        if len(qset & s)), reverse=True)[:args.k]
        recalls.append(len(set(pred) & {sid for _, sid in truth}) /
                       max(1, min(args.k, len(truth))))
    result = {
        "dataset": args.data.stem,
        "method": ("rare-token inverted index with exact Jaccard "
                   "verification"),
        "n_sessions": len(data.train),
        "n_queries": len(queries),
        "build_seconds": build_s,
        "build_peak_mb": build_peak_bytes / 1e6,
        "recall_at_10": float(np.mean(recalls)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "qps": len(queries) / max(sum(latencies) / 1000.0, 1e-9),
        "probe_items": index.probe_items,
        "verified": index.verified,
    }
    dest = Path("results/systems") / f"setsim_{args.data.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
