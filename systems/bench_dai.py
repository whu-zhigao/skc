#!/usr/bin/env python3
"""E29: DAI training acceleration — delta-aware invalidation in co-training.

Integrates the DAI strategy into the co-training loop: instead of
re-pooling ALL session profiles every epoch, only re-pool profiles whose
underlying item embeddings drifted beyond threshold θ.

Run: python3 systems/bench_dai.py --data data/diginetica.pkl --theta 0.1
Writes results/systems/dai_train_<stem>.json
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from mplignn.data import (build_adjacency, load_sessions,  # noqa: E402
                          pad_batch)
from mplignn.metrics import full_sort_metrics  # noqa: E402
from mplignn.model import MPLI_GNN  # noqa: E402
from systems.train_kv import KVRelation  # noqa: E402
from systems.igms import IGMS, IGMSConfig  # noqa: E402

REAL = Path(__file__).parent.parent / "results" / "real"


def repool_full(ig, corpus, emb):
    """Full re-pooling (R=1 baseline): all profiles recomputed."""
    t0 = time.perf_counter()
    for i, s in enumerate(corpus):
        v = emb[np.asarray(sorted(set(s)))].mean(axis=0)
        ig.values[i] = np.clip(np.round(
            v / max(np.abs(v).max() / 127.0, 1e-8)), -127, 127).astype(np.int8)
        ig._row_scale[i] = max(np.abs(v).max() / 127.0, 1e-8)
    return time.perf_counter() - t0


def repool_delta(ig, corpus, emb_old, emb_new, theta):
    """DAI re-pooling: only sessions containing items that drifted >θ."""
    t0 = time.perf_counter()
    drift = (emb_new - emb_old).norm(dim=-1) if isinstance(emb_old, torch.Tensor) \
        else np.linalg.norm(emb_new - emb_old, axis=-1)
    if isinstance(drift, torch.Tensor):
        moved = set(drift > theta)
    else:
        moved = set((drift > theta).nonzero()[0].tolist())

    # Pre-compute which sessions contain moved items
    n_recomputed = 0
    for i, s in enumerate(corpus):
        items = set(int(x) for x in s)
        if items & moved:
            v = emb_new[np.asarray(sorted(items))].mean(axis=0)
            ig.values[i] = np.clip(np.round(
                v / max(np.abs(v).max() / 127.0, 1e-8)), -127, 127).astype(np.int8)
            ig._row_scale[i] = max(np.abs(v).max() / 127.0, 1e-8)
            n_recomputed += 1
    elapsed = time.perf_counter() - t0
    return elapsed, n_recomputed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_sessions(args.data)
    stem = args.data.stem

    model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=2,
                     n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                     no_residual=True)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Build IGMS store
    print(f"[{stem}] building IGMS store...")
    t0 = time.perf_counter()
    ig = IGMS(IGMSConfig(sig_len=64, bands=32, k_r=10, max_cand=512,
                         bucket_layout="sorted", value_dim=64))
    ig.insert_batch(data.train)
    n = len(data.train)
    ig.values = np.zeros((n, 64), dtype=np.int8)
    ig._row_scale = np.ones(n, dtype=np.float32)
    emb = model.embedding.weight.detach().numpy()
    repool_full(ig, data.train, emb)
    nb = {}
    for split in ("train", "valid", "test"):
        nb[split] = [ig.search(s, 10) for s in getattr(data, split)]
    rel_train = KVRelation(ig, nb["train"])
    rel_valid = KVRelation(ig, nb["valid"])
    rel_test = KVRelation(ig, nb["test"])
    print(f"  built in {time.perf_counter()-t0:.1f}s")

    results = {"dataset": stem, "theta": args.theta, "seed": args.seed,
               "epochs": [], "summary": {}}
    total_full_s = 0.0
    total_dai_s = 0.0
    total_recomputed = 0
    total_sessions = 0

    @torch.no_grad()
    def evaluate(model, sessions, rel, batch_size=128):
        model.eval()
        agg, cnt = defaultdict(float), 0
        for i in range(0, len(sessions), batch_size):
            chunk = sessions[i:i + batch_size]
            ids = torch.arange(i, i + len(chunk))
            a_r, r0 = rel.batch_relation(ids, model.embedding.weight)
            seqs, lens, targets = pad_batch(chunk, "cpu")
            a = build_adjacency(seqs, lens)
            out = model(seqs, lens, a, a_r, r0)
            for k, v in full_sort_metrics(out["scores"], targets).items():
                agg[k] += v
            cnt += 1
        return {k: v / cnt for k, v in agg.items()}

    from collections import defaultdict

    best_mrr, best_state, best_epoch = -1.0, None, -1
    for epoch in range(1, args.epochs + 1):
        t_ep = time.perf_counter()

        # Save old embeddings for delta detection
        emb_old = model.embedding.weight.detach().clone()

        # Train
        model.train()
        perm = np.random.permutation(n)
        limit = min(len(perm), 300 * 128)
        tot, nb_batches = 0.0, 0
        for i in range(0, limit, 128):
            ids = torch.from_numpy(perm[i:i + 128])
            chunk = [data.train[j] for j in ids.tolist()]
            seqs, lens, targets = pad_batch(chunk, "cpu")
            a = build_adjacency(seqs, lens)
            a_r, r0 = rel_train.batch_relation(ids, model.embedding.weight)
            out = model(seqs, lens, a, a_r, r0)
            loss = F.cross_entropy(out["scores"], targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb_batches += 1

        # Measure both re-pooling strategies
        emb_new = model.embedding.weight.detach().numpy()
        t_full = repool_full(ig, data.train[:1000], emb_new) \
            * n / 1000  # extrapolate
        t_dai, n_recomputed = repool_delta(
            ig, data.train, emb_old.numpy(), emb_new, args.theta)

        # Apply DAI (the actual strategy used in training)
        emb_final = model.embedding.weight.detach().numpy()
        t_dai, n_recomputed = repool_delta(
            ig, data.train, emb_old.numpy(), emb_final, args.theta)

        total_full_s += t_full
        total_dai_s += t_dai
        total_recomputed += n_recomputed
        total_sessions += n

        vm = evaluate(model, data.valid, rel_valid)
        epoch_s = time.perf_counter() - t_ep
        results["epochs"].append({
            "epoch": epoch, "loss": tot / max(nb_batches, 1),
            "val_mrr": vm["MRR@20"], "val_hr": vm["HR@20"],
            "epoch_s": epoch_s,
            "repool_full_s": t_full, "repool_dai_s": t_dai,
            "dai_recomputed": n_recomputed, "dai_skipped": n - n_recomputed,
        })
        print(f"  ep {epoch:2d} loss {tot/max(nb_batches,1):.3f} "
              f"val MRR {vm['MRR@20']:.4f} "
              f"DAI {t_dai:.1f}s (skip {100*(n-n_recomputed)/n:.0f}%) "
              f"vs full {t_full:.1f}s")
        if vm["MRR@20"] > best_mrr:
            best_mrr, best_epoch = vm["MRR@20"], epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test = evaluate(model, data.test, rel_test)
    results["summary"] = {
        "test_hr": 100 * test["HR@20"], "test_mrr": 100 * test["MRR@20"],
        "best_epoch": best_epoch,
        "total_full_repool_s": total_full_s,
        "total_dai_repool_s": total_dai_s,
        "speedup": total_full_s / max(total_dai_s, 1e-9),
        "avg_skip_rate": 1 - total_recomputed / max(total_sessions, 1),
    }
    print(f"\n[{stem}] DAI co-training: test {100*test['HR@20']:.2f}/"
          f"{100*test['MRR@20']:.2f}")
    print(f"  repool: full {total_full_s:.1f}s vs DAI {total_dai_s:.1f}s "
          f"({total_full_s/max(total_dai_s,1e-9):.1f}× speedup)")
    print(f"  avg skip rate: {100*results['summary']['avg_skip_rate']:.1f}%")

    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"dai_train_{stem}_theta{args.theta}.json"
    f.write_text(json.dumps(results, indent=2))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
