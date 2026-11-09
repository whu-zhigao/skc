#!/usr/bin/env python3
"""Systems benchmark E3: KV-ized global memory at the serving path.

Replaces the batch-time relation assembly of the reference system

    neighbor item ids  ->  gather embeddings  ->  mean  ->  cosine (a_r)

with a key-value read

    session key  ->  band probe  ->  dequantized int8 profile (r0)  ->  cosine

Three evaluations on the released instrument checkpoint (frozen weights):

  A. baseline assembly (dynamic gather over item ids)      -> HR/MRR
  B. KV values pooled with the SAME (final) table          -> HR/MRR
     (isolates quantization + neighbor-set change)
  C. KV values pooled with a random table                  -> HR/MRR
     (upper bound on embedding-drift staleness)

plus per-batch assembly latency, memory footprint, and neighbor-set
recall/overlap diagnostics.

Run: python3 systems/bench_kv_accuracy.py --data data/diginetica.pkl
Writes results/systems/kv_accuracy_<stem>.json
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from mplignn.data import (FastRelationIndex, build_adjacency,  # noqa: E402
                          load_sessions, pad_batch)
from mplignn.model import MPLI_GNN  # noqa: E402
from systems.igms import IGMS, IGMSConfig  # noqa: E402

REAL = Path(__file__).parent.parent / "results" / "real"


@torch.no_grad()
def evaluate(model, sessions, relation, device: str):
    """Evaluate next-item quality with an explicit prefix-only query."""
    scores, targets = [], []
    for start in range(0, len(sessions), 128):
        chunk = sessions[start:start + 128]
        seqs, lens, target = pad_batch(chunk, device)
        adjacency = build_adjacency(seqs, lens)
        ids = torch.arange(start, start + len(chunk), device=seqs.device)
        a_r, r0 = relation.batch_relation(ids, model.embedding.weight)
        out = model(seqs, lens, adjacency, a_r, r0)
        out = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(out, dict):
            out = out["scores"]
        scores.append(out.float().cpu().numpy())
        targets.append(target.cpu().numpy())
    logits = np.concatenate(scores)
    target = np.concatenate(targets)
    order = np.argsort(-logits, axis=1)
    ranks = np.array([np.where(order[i] == target[i])[0][0] + 1
                      for i in range(len(target))])
    return {
        "HR@20": float((ranks <= 20).mean()),
        "MRR@20": float((1.0 / ranks)[ranks <= 20].mean()),
    }


class KVRelation:
    """Duck-typed drop-in for FastRelationIndex at the serving path."""

    def __init__(self, ig: IGMS, neighbors):
        self.ig = ig
        self.neigh = torch.as_tensor(np.asarray(neighbors, dtype=np.int64))

    def batch_relation(self, session_ids, emb_table):
        nb = self.neigh[session_ids.cpu()]                      # (B, K)
        prof = self.ig.profile(nb.reshape(-1).numpy())          # (B*K, d)
        r0 = torch.as_tensor(prof, dtype=torch.float32).view(
            nb.shape[0], nb.shape[1], -1)
        a = torch.nn.functional.cosine_similarity(
            r0.unsqueeze(2), r0.unsqueeze(1), dim=-1)
        eye = torch.eye(a.shape[-1]).unsqueeze(0)
        a = a * (1 - eye)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
        return a, r0


def pooled_profiles(corpus, emb: np.ndarray) -> np.ndarray:
    out = np.zeros((len(corpus), emb.shape[1]), dtype=np.float32)
    for i, s in enumerate(corpus):
        out[i] = emb[np.asarray(sorted(set(s)))].mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=REAL / "abl_no_residual_diginetica.pt")
    ap.add_argument("--max-query", type=int, default=1000)
    args = ap.parse_args()
    data = load_sessions(args.data)
    data.test = data.test[:min(args.max_query, len(data.test))]
    stem = args.data.stem
    k = 10

    model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=2,
                     n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                     no_residual=True)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu",
                                     weights_only=True))
    model.eval()
    emb = model.embedding.weight.detach().numpy()

    # ---- A. baseline assembly (relcache hit)
    query_prefixes = [s[:-1] if len(s) > 1 else s for s in data.test]
    rel = FastRelationIndex(query_prefixes, data.train, data.n_items, k,
                            cache=None)
    a_base = evaluate(model, data.test, rel, "cpu")
    hr_base, mrr_base = 100 * a_base["HR@20"], 100 * a_base["MRR@20"]

    # ---- IGMS with values pooled from the FINAL table
    t0 = time.perf_counter()
    ig = IGMS(IGMSConfig(sig_len=64, bands=32, k_r=k, max_cand=512,
                         bucket_layout="sorted"))
    vals = pooled_profiles(data.train, emb)
    ig.insert_batch(data.train, pooled=vals)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    neighbors = [ig.search(s, k) for s in query_prefixes]
    search_s = time.perf_counter() - t0
    kv = KVRelation(ig, neighbors)
    a_kv = evaluate(model, data.test, kv, "cpu")
    hr_kv, mrr_kv = 100 * a_kv["HR@20"], 100 * a_kv["MRR@20"]

    # ---- C. staleness bound: values pooled from a random table
    torch.manual_seed(0)
    rand_emb = torch.nn.Embedding(data.n_items, 64)
    torch.nn.init.normal_(rand_emb.weight, std=0.02)
    ig2 = IGMS(IGMSConfig(sig_len=64, bands=32, k_r=k, max_cand=512,
                          bucket_layout="sorted"))
    ig2.insert_batch(data.train,
                     pooled_profiles(data.train,
                                     rand_emb.weight.detach().numpy()))
    kv2 = KVRelation(ig2, neighbors)      # same neighbor sets (same keys)
    a_st = evaluate(model, data.test, kv2, "cpu")
    hr_st, mrr_st = 100 * a_st["HR@20"], 100 * a_st["MRR@20"]

    # ---- 2x2 ablation: disentangle neighbor-set change from value path.
    # Cell A2b: IGMS neighbors + DYNAMIC gather (item-id semantics kept).
    from systems.bench_retrieval import baseline_neighbor_ids

    class DynamicFromIds:
        def __init__(self, neighbors, corpus):
            self.neigh = torch.as_tensor(np.asarray(neighbors, dtype=np.int64))
            self.corpus = corpus

        def batch_relation(self, session_ids, emb_table):
            nb = self.neigh[session_ids.cpu()].numpy()
            r0 = torch.stack([
                torch.stack([emb_table[torch.as_tensor(
                    sorted(set(self.corpus[j]))[:12], dtype=torch.long)]
                    .mean(dim=0) for j in row]) for row in nb])
            a = torch.nn.functional.cosine_similarity(
                r0.unsqueeze(2), r0.unsqueeze(1), dim=-1)
            eye = torch.eye(a.shape[-1]).unsqueeze(0)
            a = a * (1 - eye)
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
            return a, r0

    dyn = DynamicFromIds(neighbors, data.train)
    a_dyn = evaluate(model, data.test, dyn, "cpu")
    # Cell A2c: BASELINE neighbors + KV values (quantization isolated).
    base_nb = baseline_neighbor_ids([s[:-1] if len(s) > 1 else s for s in data.test], data.train, k)
    kv_base_nb = KVRelation(ig, base_nb)
    a_q = evaluate(model, data.test, kv_base_nb, "cpu")

    # ---- assembly latency (per 128-session batch)
    ids = torch.arange(0, 6 * 128)
    for _ in range(2):                                   # warmup
        rel.batch_relation(ids[:128], model.embedding.weight)
        kv.batch_relation(ids[:128], model.embedding.weight)
    tb, tk = [], []
    for i in range(0, len(ids), 128):
        t0 = time.perf_counter()
        rel.batch_relation(ids[i:i + 128], model.embedding.weight)
        tb.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        kv.batch_relation(ids[i:i + 128], model.embedding.weight)
        tk.append(time.perf_counter() - t0)

    # ---- neighbor diagnostics + memory
    base_ids = rel.neigh_items                             # (m, K, L) items
    kv_mem = ig.memory_bytes()
    res = {
        "dataset": stem, "ckpt": args.ckpt.name, "k": k,
        "n_query": len(data.test), "query_protocol": "prefix-only next item",
        "baseline": {"HR@20": hr_base, "MRR@20": mrr_base,
                     "assembly_ms_per_batch": 1000 * st.mean(tb),
                     "neighbor_store_mb":
                         base_ids.element_size() * base_ids.nelement() / 1e6},
        "kv_final_table": {"HR@20": hr_kv, "MRR@20": mrr_kv,
                           "hr_delta": hr_kv - hr_base,
                           "mrr_delta": mrr_kv - mrr_base,
                           "assembly_ms_per_batch": 1000 * st.mean(tk),
                           "build_s": build_s, "search_all_test_s": search_s,
                           "store_mb": {kk: vv / 1e6
                                        for kk, vv in kv_mem.items()}},
        "ablation_2x2": {
            "igms_neighbors_dynamic_values": {
                "HR@20": 100 * a_dyn["HR@20"], "MRR@20": 100 * a_dyn["MRR@20"],
                "note": "neighbor-set change isolated; gather semantics kept"},
            "baseline_neighbors_kv_values": {
                "HR@20": 100 * a_q["HR@20"], "MRR@20": 100 * a_q["MRR@20"],
                "note": "value path isolated; neighbor sets as trained"}},
        "kv_stale_bound": {"HR@20": hr_st, "MRR@20": mrr_st,
                           "hr_delta": hr_st - hr_base,
                           "mrr_delta": mrr_st - mrr_base,
                           "note": "values pooled from a random table; "
                                   "upper bound on embedding-drift staleness"},
    }
    print(f"[{stem}] baseline HR/MRR {hr_base:.2f}/{mrr_base:.2f} "
          f"({1000*st.mean(tb):.2f} ms/batch)")
    print(f"[{stem}] KV(final)     {hr_kv:.2f}/{mrr_kv:.2f} "
          f"(dHR {hr_kv-hr_base:+.2f}) ({1000*st.mean(tk):.2f} ms/batch)")
    print(f"[{stem}] KV(stale-bnd) {hr_st:.2f}/{mrr_st:.2f} "
          f"(dHR {hr_st-hr_base:+.2f})")
    print(f"[{stem}] IGMS-nb + dynamic values  "
          f"{100*a_dyn['HR@20']:.2f}/{100*a_dyn['MRR@20']:.2f}")
    print(f"[{stem}] base-nb + KV values      "
          f"{100*a_q['HR@20']:.2f}/{100*a_q['MRR@20']:.2f}")
    dest = Path("results/systems")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"kv_accuracy_{stem}.json"
    f.write_text(json.dumps(res, indent=2))
    print("wrote", f)


if __name__ == "__main__":
    main()
