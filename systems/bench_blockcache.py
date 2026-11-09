#!/usr/bin/env python3
"""Systems benchmark E6: cross-batch affinity block cache (eval path).

The first stacked block of the adaptive global encoder consumes the
batch-unique item embeddings and computes pairwise tanh affinities.
At serving time (frozen weights) these affinities depend only on the
item PAIR, not the batch: a group-block cache (item space partitioned
into groups of G) turns repeated co-occurrence into exact reuse.
Deeper stacked blocks see batch-dependent inputs and are excluded.

Measured: exactness (max |diff| vs uncached forward), hit rate, eval
wall time with/without cache, cache footprint.

Run: python3 systems/bench_blockcache.py --data data/diginetica.pkl
Writes results/systems/blockcache_<stem>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from mplignn.data import (FastRelationIndex, build_adjacency,  # noqa: E402
                          load_sessions, pad_batch)
from mplignn.model import MPLI_GNN  # noqa: E402

REAL = Path(__file__).parent.parent / "results" / "real"


class CachedGlobalEncoder(torch.nn.Module):
    """Wraps a trained GlobalEncoder; caches first-block score blocks
    keyed by (group_i, group_j). Weights are shared, not copied.

    The reference model calls ``self.global_(x)`` without item ids; the
    harness therefore sets ``pending_uniq`` (the unique-id tensor whose
    rows align with x) immediately before each forward. Explicit ids
    may also be passed for direct calls."""

    def __init__(self, inner, n_items, group=128):
        super().__init__()
        self.inner = inner
        self.group = group
        self.n_groups = (n_items + group - 1) // group
        self.cache = {}
        self.hits = self.misses = 0
        self.hits_flops = self.total_flops = 0
        self.pending_uniq = None
        self.emb_table = None

    def _block(self, gi, gj):
        """Full fixed-shape (h, G, G) score block for a group pair,
        computed from the embedding table on miss."""
        key = (gi, gj) if gi <= gj else (gj, gi)
        blk = self.cache.get(key)
        if blk is None:
            self.misses += 1
            G = self.group
            ei = self.emb_table[gi * G:(gi + 1) * G]
            ej = self.emb_table[gj * G:(gj + 1) * G]
            q = self.inner.wq(ei).view(len(ei), self.inner.h, self.inner.dk)
            k = self.inner.wk(ej).view(len(ej), self.inner.h, self.inner.dk)
            blk = torch.tanh(torch.einsum("ihd,jhd->hij", q, k)
                             / self.inner.dk ** 0.5)
            if key[0] != key[1]:
                blk_T = blk if key == (gi, gj) else blk.transpose(-2, -1).contiguous()
                self.cache[key] = blk
                return blk if gi <= gj else blk.transpose(-2, -1)
            self.cache[key] = blk
            return blk
        self.hits += 1
        self.hits_flops += self.group * self.group * self.inner.dk * self.inner.h
        return blk if key == (gi, gj) else blk.transpose(-2, -1)

    def forward(self, x, ids=None):
        inner = self.inner
        if ids is None:
            ids = self.pending_uniq
            assert ids is not None and len(ids) == x.shape[0]
        if self.emb_table is None:
            self.emb_table = x.new_empty(0)   # set by harness post-load
        n = x.shape[0]
        g = ids // self.group
        v = inner.wv(x).view(n, inner.h, inner.dk).transpose(0, 1)
        scores = torch.empty(inner.h, n, n)
        for gi in torch.unique(g):
            rows_i = (g == gi).nonzero(as_tuple=True)[0]
            li = ids[rows_i] % self.group
            for gj in torch.unique(g):
                rows_j = (g == gj).nonzero(as_tuple=True)[0]
                lj = ids[rows_j] % self.group
                self.total_flops += len(rows_i) * len(rows_j) * inner.dk * inner.h
                self.miss_probe = True
                blk = self._block(int(gi), int(gj))
                scores[:, rows_i.unsqueeze(1), rows_j.unsqueeze(0)] = \
                    blk[:, li.unsqueeze(1), lj.unsqueeze(0)]
        out = scores @ v
        out = out.transpose(0, 1).reshape(n, inner.d)
        inp = inner.drop(inner.wo(out)) + x
        for _ in range(inner.n_blocks_m - 1):
            inp = inner._attend(inp) + inp
        return inp

        return inner(x)


@torch.no_grad()
def run_eval(model, data, rel, cached_model=None, batch_size=128):
    """One pass over the test set; returns (seconds, HR@20, MRR@20).
    The cached run swaps model.global_ for the wrapper and feeds it the
    batch's unique ids via pending_uniq before each forward."""
    from mplignn.metrics import full_sort_metrics
    saved = model.global_
    if cached_model is not None:
        model.global_ = cached_model
    hits = []
    t0 = time.perf_counter()
    for i in range(0, len(data.test), batch_size):
        chunk = data.test[i:i + batch_size]
        ids = torch.arange(i, i + len(chunk))
        a_r, r0 = rel.batch_relation(ids, model.embedding.weight)
        seqs, lens, targets = pad_batch(chunk, "cpu")
        a = build_adjacency(seqs, lens)
        if cached_model is not None:
            uniq = torch.unique(seqs)
            cached_model.pending_uniq = uniq
        out = model(seqs, lens, a, a_r, r0)
        m = full_sort_metrics(out["scores"], targets)
        hits.append((m["HR@20"], m["MRR@20"]))
    sec = time.perf_counter() - t0
    model.global_ = saved
    n = len(hits)
    return sec, 100 * sum(h[0] for h in hits) / n, \
        100 * sum(h[1] for h in hits) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=REAL / "abl_no_residual_diginetica.pt")
    ap.add_argument("--group", type=int, default=128)
    args = ap.parse_args()
    data = load_sessions(args.data)
    stem = args.data.stem

    model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=2,
                     n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                     no_residual=True)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu",
                                     weights_only=True))
    model.eval()
    rel = FastRelationIndex(data.test, data.train, data.n_items, 10,
                            cache=args.data.parent / "relcache"
                            / f"{stem}_test.npz")

    # exactness probe: END-TO-END on one batch (swap model.global_,
    # feed pending_uniq, compare full-model scores)
    cached = CachedGlobalEncoder(model.global_, data.n_items, args.group)
    cached.emb_table = model.embedding.weight.detach()
    chunk = data.test[:128]
    with torch.no_grad():
        seqs0, lens0, targets0 = pad_batch(chunk, "cpu")
        a0 = build_adjacency(seqs0, lens0)
        a_r0, r00 = rel.batch_relation(torch.arange(128),
                                       model.embedding.weight)
        saved0 = model.global_
        ref_scores = model(seqs0, lens0, a0, a_r0, r00)["scores"]
        model.global_ = cached
        cached.pending_uniq = torch.unique(seqs0)
        got_scores = model(seqs0, lens0, a0, a_r0, r00)["scores"]
        model.global_ = saved0
    maxdiff = (ref_scores - got_scores).abs().max().item()

    # full eval, uncached vs cached (cache pre-warmed above continues)
    t_ref, hr_ref, mrr_ref = run_eval(model, data, rel, cached_model=None)
    t_cac, hr_cac, mrr_cac = run_eval(model, data, rel, cached_model=cached)
    hit_rate = cached.hits / max(1, cached.hits + cached.misses)
    flop_rate = cached.hits_flops / max(1, cached.total_flops)
    cache_bytes = sum(t.nbytes for t in cached.cache.values())

    res = {"dataset": stem, "group": args.group,
           "exactness_max_abs_diff": maxdiff,
           "eval_uncached_s": t_ref, "eval_cached_s": t_cac,
           "speedup": t_ref / t_cac,
           "hr_uncached": hr_ref, "hr_cached": hr_cac,
           "block_hit_rate": hit_rate,
           "affinity_flops_served_from_cache": flop_rate,
           "cache_mb": cache_bytes / 1e6,
           "n_cached_blocks": len(cached.cache)}
    print(json.dumps(res, indent=2))
    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"blockcache_{stem}.json"
    f.write_text(json.dumps(res, indent=2))
    print("wrote", f)


if __name__ == "__main__":
    main()
