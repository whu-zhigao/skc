#!/usr/bin/env python3
"""Co-training experiment: train the instrument WITH the IGMS store.

The serving-side 2x2 ablation showed that the KV path is accuracy-neutral
but the retrieval-semantics change is not (zero-shot +5.5 HR on a frozen
checkpoint trained with the old heuristic). The clean systems protocol is
to co-train with the store so train and serve share one retrieval
semantic, with a value-refresh schedule (R epochs) controlling write-time
staleness during training.

Configs (budget protocol, identical to the released instrument runs):
  --refresh 1   values re-pooled from current embeddings every epoch
  --refresh 0   values frozen at initialization (staleness upper bound)
  --baseline    plain run with FastRelationIndex (reference)

Run: python3 systems/train_kv.py --data data/diginetica.pkl --refresh 1
Writes results/systems/cotrain_<stem>_r<refresh>.json (+ .pt)
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
from mplignn.data import (FastRelationIndex, build_adjacency,  # noqa: E402
                          load_sessions, pad_batch)
from mplignn.model import MPLI_GNN  # noqa: E402
from scripts.train import evaluate  # noqa: E402
from systems.igms import IGMS, IGMSConfig  # noqa: E402


class KVRelation:
    def __init__(self, ig, neighbors):
        self.ig = ig
        self.neigh = torch.as_tensor(np.asarray(neighbors, dtype=np.int64))

    def batch_relation(self, session_ids, emb_table):
        nb = self.neigh[session_ids.cpu()]
        prof = self.ig.profile(nb.reshape(-1).numpy())
        r0 = torch.as_tensor(prof, dtype=torch.float32).view(*nb.shape, -1)
        a = torch.nn.functional.cosine_similarity(
            r0.unsqueeze(2), r0.unsqueeze(1), dim=-1)
        eye = torch.eye(a.shape[-1]).unsqueeze(0)
        a = a * (1 - eye)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
        return a, r0


def repool(ig, corpus, emb, ids_slice=None):
    for i, s in enumerate(corpus):
        v = emb[np.asarray(sorted(set(s)))].mean(axis=0)
        ig.values[i] = np.clip(np.round(
            v / max(np.abs(v).max() / 127.0, 1e-8)), -127, 127).astype(np.int8)
        ig._row_scale[i] = max(np.abs(v).max() / 127.0, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--refresh", type=int, default=1)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_sessions(args.data)
    stem = args.data.stem
    device = "cpu"

    model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=2,
                     n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                     no_residual=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.baseline:
        rc = args.data.parent / "relcache"
        rel_train = FastRelationIndex(data.train, data.train,
                                      data.n_items, 10,
                                      cache=rc / f"{stem}_train.npz")
        rel_valid = FastRelationIndex(data.valid, data.train,
                                      data.n_items, 10,
                                      cache=rc / f"{stem}_valid.npz")
        rel_test = FastRelationIndex(data.test, data.train,
                                     data.n_items, 10,
                                     cache=rc / f"{stem}_test.npz")
        ig = None
    else:
        t0 = time.time()
        ig = IGMS(IGMSConfig(sig_len=64, bands=32, k_r=10, max_cand=512,
                             value_dim=64))
        ig.insert_batch(data.train)
        n = len(data.train)
        ig.values = np.zeros((n, 64), dtype=np.int8)
        ig._row_scale = np.ones(n, dtype=np.float32)
        emb = model.embedding.weight.detach().numpy()
        repool(ig, data.train, emb)
        nb = {}
        for split in ("train", "valid", "test"):
            nb[split] = [ig.search(s, 10)
                         for s in getattr(data, split)]
        rel_train = KVRelation(ig, nb["train"])
        rel_valid = KVRelation(ig, nb["valid"])
        rel_test = KVRelation(ig, nb["test"])
        print(f"store built in {time.time()-t0:.1f}s "
              f"(+neighbors for {len(nb['train'])}+{len(nb['valid'])}"
              f"+{len(nb['test'])} anchors)")

    best_mrr, best_state, best_epoch, idle = -1.0, None, -1, 0
    history, epoch_seconds = [], []
    t_start = time.time()
    n_train = len(data.train)
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        if ig is not None and args.refresh > 0 and epoch > 1 \
                and (epoch - 1) % args.refresh == 0:
            repool(ig, data.train, model.embedding.weight.detach().numpy())
        model.train()
        perm = np.random.permutation(n_train)
        limit = min(len(perm), args.max_batches * args.batch_size)
        tot, nb_ = 0.0, 0
        for i in range(0, limit, args.batch_size):
            ids = torch.from_numpy(perm[i:i + args.batch_size])
            chunk = [data.train[j] for j in ids.tolist()]
            seqs, lens, targets = pad_batch(chunk, device)
            a = build_adjacency(seqs, lens)
            a_r, r0 = rel_train.batch_relation(ids, model.embedding.weight)
            out = model(seqs, lens, a, a_r, r0)
            loss = F.cross_entropy(out["scores"], targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb_ += 1
        vm = evaluate(model, data.valid, rel_valid, device, args.batch_size)
        epoch_seconds.append(round(time.time() - t_ep, 2))
        history.append({"epoch": epoch, "loss": tot / max(nb_, 1),
                        "valid": vm})
        print(f"epoch {epoch:3d} loss {tot/max(nb_,1):.4f} "
              f"val MRR@20 {vm['MRR@20']:.4f} HR@20 {vm['HR@20']:.4f} "
              f"({time.time()-t_start:.0f}s)", flush=True)
        if vm["MRR@20"] > best_mrr:
            best_mrr, best_epoch, idle = vm["MRR@20"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            idle += 1
            if idle >= args.patience:
                break

    model.load_state_dict(best_state)
    t0 = time.perf_counter()
    test = evaluate(model, data.test, rel_test, device, args.batch_size)
    test_s = time.perf_counter() - t0
    res = {
        "variant": "baseline" if args.baseline else f"cotrain_r{args.refresh}",
        "dataset": stem, "seed": args.seed,
        "test": test, "best_epoch": best_epoch,
        "wall_seconds": time.time() - t_start,
        "epoch_seconds": epoch_seconds, "history": history,
        "eval_test_seconds": test_s,
    }
    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    tag = "baseline" if args.baseline else f"r{args.refresh}"
    f = dest / f"cotrain_{stem}_{tag}_s{args.seed}.json"
    f.write_text(json.dumps(res, indent=2))
    torch.save(best_state, f.with_suffix(".pt"))
    print(f"[{stem}] {res['variant']} test HR/MRR "
          f"{100*test['HR@20']:.2f}/{100*test['MRR@20']:.2f} "
          f"best_epoch {best_epoch} -> {f}")


if __name__ == "__main__":
    main()
