#!/usr/bin/env python3
"""D2: Yoochoose-1/4 scaling study -- the suite's third dataset
(data/yoochoose_quarter.pkl, 1.7M train sessions / 23k items;
checkpoint mplignn_yoochoose_quarter.pt via the same resolver chain the
driver uses). Three sections, each mirroring its parent stage and
writing independent per-config JSONs under results/systems/d2/:

  workload: A1-style neighbor-graph reuse characterization (corpus
            size, active fraction, Gini, top-m coverage, hub stability).
  pareto:   B1-style Tier-1 hit-rate vs memory-budget Pareto with the
            SAME admission family (lru/lfu/arc/tinylfu/wtinylfu/lecar/
            greedydualsize + nfa sketch/oracle) and the exact
            full-cache reference point. Reuses exp_b1 machinery.
  ann:      D1-style ANN baselines on the pooled-profile vectors:
            MinHash-LSH vs exact cosine/Jaccard ground truths (+ FAISS
            when installed) and the exact full-scan reference row.

Sections are independently idempotent (--sections picks a subset). The
ANN section indexes a corpus subsample (--ann-corpus-cap, recorded in
the JSON) to bound build time on this 1.7M-session corpus."""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ADMISSION_CITATIONS, REAL, config_out, data_file,
 serving_prefixes,
                    gini, jaccard_topk, load_data, load_model,
                    neighbor_ids, pool_profiles, write_json)
from exp_a1_workload import topm_overlap
from exp_b1_cache_pareto import (ALL_POLICIES, stream_hits_online,
                                 stream_nfa_sketch)
from exp_d1_ann_baselines import MinHashLSH, rss_mb


def resolve_ckpt(stem):
    """Yoochoose checkpoint fallback chain (results/real/): the
    abl_no_residual_<stem>.pt reference if present, else the full
    mplignn_<stem>.pt (same architecture, per-dataset item count)."""
    for name in (f"abl_no_residual_{stem}.pt", f"mplignn_{stem}.pt",
                 f"full_mplignn_{stem}.pt"):
        p = REAL / name
        if p.exists():
            return p
    return REAL / f"mplignn_{stem}.pt"


# --------------------------------------------------------------- workload
def run_workload(data, queries, nbrs, k, n_windows, topm):
    freq = Counter(sid for nb in nbrs for sid in nb)
    total = sum(len(nb) for nb in nbrs)
    m = np.array(sorted(freq.values(), reverse=True))
    coverage = {str(frac): float(
        m[:max(1, int(len(m) * frac))].sum() / m.sum())
        for frac in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)}
    nq = len(nbrs)
    W = n_windows
    edges = np.linspace(0, nq, W + 1, dtype=int)
    hub_sets = []
    for w in range(W):
        wf = Counter(sid for nb in nbrs[edges[w]:edges[w + 1]]
                     for sid in nb)
        hub_sets.append({sid for sid, _ in wf.most_common(topm)})
    stab = [topm_overlap(hub_sets[i], hub_sets[i + 1])
            for i in range(W - 1)]
    global_hubs = {sid for sid, _ in freq.most_common(topm)}
    glob = [topm_overlap(global_hubs, h) for h in hub_sets]
    return {
        "corpus_size": len(data.train),
        "n_query": nq, "k": k,
        "total_lookups": total,
        "active_fraction": len(freq) / len(data.train),
        "unique_neighbor_sessions": len(freq),
        "gini": gini(freq.values()),
        "top_coverage": coverage,
        "hub_stability": {
            "windows": W, "topm": topm,
            "consecutive_jaccard": stab,
            "mean_consecutive_jaccard": float(np.mean(stab)),
            "global_vs_window_jaccard": glob,
            "mean_global_vs_window": float(np.mean(glob))},
        "note": "A1-style workload characterization on Yoochoose-1/4 "
                "(1.7M-session corpus; same measurement code as "
                "exp_a1_workload)"}


# ----------------------------------------------------------------- pareto
def run_pareto(nbrs, d, seed, sketch_w, sketch_d):
    freq = Counter(sid for nb in nbrs for sid in nb)
    total = max(1, sum(len(nb) for nb in nbrs))
    budgets_mb = [1, 2, 5, 10, 20, 50, 100]
    rows = []
    for mb in budgets_mb:
        cap = int(mb * 1e6 / (d * 4))
        row = {"budget_mb": mb, "budget_sessions": cap}
        top = {sid for sid, _ in freq.most_common(cap)}
        row["oracle"] = row["nfa_offline"] = sum(
            1 for nb in nbrs for s in nb if s in top) / total
        for pol in ALL_POLICIES:
            row[pol] = stream_hits_online(nbrs, cap, pol, seed=seed)
        row["nfa_sketch"], row["sketch_bytes"] = stream_nfa_sketch(
            nbrs, cap, sketch_w, sketch_d)
        row["meta_bytes_session_level"] = cap * 24
        rows.append(row)
        print(f"  {mb:3d}MB cap={cap:7d}: " + " ".join(
            f"{kk}={vv:.3f}" for kk, vv in row.items()
            if isinstance(vv, float)))
    return {
        "n_query": len(nbrs), "budgets": rows,
        "active_unique": len(freq),
        "admissions": {p: ADMISSION_CITATIONS.get(p, p)
                       for p in ALL_POLICIES},
        "seed": seed,
        "exact_reference": {
            "budget_sessions": len(freq),
            "budget_mb": round(len(freq) * d * 4 / 1e6, 3),
            "hit": 1.0,
            "note": "exact full-cache reference (all active neighbors "
                    "materialized): every lookup hits by construction"},
        "note": "B1-style Tier-1 admission Pareto on Yoochoose-1/4 "
                "(same replay code as exp_b1_cache_pareto)"}


# -------------------------------------------------------------------- ann
def run_ann(data, queries, k, ckpt, corpus_cap):
    corpus = (data.train if corpus_cap <= 0
              else data.train[:min(corpus_cap, len(data.train))])
    model = load_model(data, ckpt, device="cpu")
    emb = model.embedding.weight.detach().numpy()
    P = pool_profiles(emb, corpus)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    nq = len(queries)

    import torch
    qv = pool_profiles(emb, queries)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    cos_gt = torch.topk(torch.from_numpy(qv) @ torch.from_numpy(Pn).T,
                        k, dim=1).indices.numpy()
    t0 = time.perf_counter()
    jac_gt = jaccard_topk(queries, corpus, k)
    t_jac = time.perf_counter() - t0

    methods = {"exact": {
        "build_s": 0.0, "query_s": t_jac,
        "queries_per_s": nq / max(t_jac, 1e-9),
        "recall_jaccard": 1.0,
        "recall_cos": float(np.mean([len(set(p) & set(g)) / k
                                     for p, g in zip(jac_gt, cos_gt)])),
        "rss_mb_after_build": rss_mb(),
        "note": "exact full-scan Jaccard top-k (reference upper "
                "bound; the ground-truth scan itself is the timing)"}}
    common = {"corpus_size": len(corpus),
              "corpus_cap": corpus_cap, "n_query": nq, "k": k,
              "ckpt": str(ckpt), "n_items": data.n_items}

    # MinHash-LSH (stand-in for IGMS postings)
    t0 = time.perf_counter()
    lsh = MinHashLSH(corpus, r=2, b=64, seed=0)
    lsh.build()
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = [lsh.query(s, k) for s in queries]
    t_query = time.perf_counter() - t0
    methods["minhash_lsh"] = {
        "build_s": t_build, "query_s": t_query,
        "queries_per_s": nq / max(t_query, 1e-9),
        "recall_cos": float(np.mean([len(set(p) & set(g)) / k
                                     for p, g in zip(pred, cos_gt)])),
        "recall_jaccard": float(np.mean(
            [len(set(p) & set(g)) / k for p, g in zip(pred, jac_gt)])),
        "rss_mb_after_build": rss_mb()}

    try:
        import faiss
        dd = P.shape[1]
        t0 = time.perf_counter()
        quant = faiss.IndexFlatIP(dd)
        ivf = faiss.IndexIVFFlat(quant, dd, 256)
        ivf.train(Pn.astype(np.float32))
        ivf.add(Pn.astype(np.float32))
        methods["faiss_ivf"] = {"build_s": time.perf_counter() - t0}
        ivf.nprobe = 16
        _, I = ivf.search(qv.astype(np.float32), k)
        methods["faiss_ivf"].update({
            "recall_cos": float(np.mean(
                [len(set(I[i]) & set(cos_gt[i])) / k for i in range(nq)])),
            "recall_jaccard": float(np.mean(
                [len(set(I[i]) & set(jac_gt[i])) / k for i in range(nq)])),
            "rss_mb_after_build": rss_mb()})
    except ImportError:
        methods["faiss"] = "skipped: pip install faiss-cpu"

    return common, methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None,
                    help="yoochoose pickle (default: common.data_file"
                         "('yoochoose_quarter'))")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="yoochoose checkpoint (default: resolver chain"
                         " abl_no_residual_<stem> -> mplignn_<stem> ->"
                         " full_mplignn_<stem>)")
    ap.add_argument("--sections", default="workload,pareto,ann",
                    help="comma list of {workload,pareto,ann}")
    ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-windows", type=int, default=10)
    ap.add_argument("--topm", type=int, default=2000)
    ap.add_argument("--ann-corpus-cap", type=int, default=300000,
                    help="ANN corpus subsample (0 = full 1.7M; "
                         "recorded in the JSON)")
    ap.add_argument("--sketch-w", type=int, default=4096)
    ap.add_argument("--sketch-d", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun sections whose per-config JSON exists")
    args = ap.parse_args()

    args.data = args.data or data_file("yoochoose_quarter")
    if not args.data.exists():
        sys.exit(f"[d2] dataset not found: {args.data}\n"
                 f"       expected yoochoose_quarter.pkl under "
                 f"SKC_DATA_ROOT or the repo code/data/")
    stem = args.data.stem
    ckpt = args.ckpt or resolve_ckpt(stem)
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    bad = [s for s in sections
           if s not in ("workload", "pareto", "ann")]
    if bad:
        sys.exit(f"[d2] unknown sections: {bad}")

    tags = {
        "workload": f"{stem}_workload_nq{args.n_query}_k{args.k}"
                    f"_w{args.n_windows}",
        "pareto": f"{stem}_pareto_nq{args.n_query}_k{args.k}"
                  f"_d{args.d}_s{args.seed}",
        "ann": f"{stem}_ann_nq{args.n_query}_k{args.k}"
               f"_cap{args.ann_corpus_cap}",
    }
    def section_pending(s):
        """ann's per-config files are per-method; minhash_lsh (always
        written when a ckpt exists) is the section sentinel."""
        if args.out or args.force:
            return True
        tag = f"{tags[s]}_minhash_lsh" if s == "ann" else tags[s]
        _, skip = config_out("d2", tag, None, False)
        return not skip

    todo = [s for s in sections if section_pending(s)]
    if not todo:
        print("[d2] all sections exist, nothing to do (--force to rerun)")
        return

    data = load_data(args.data)
    nq = min(args.n_query, len(data.test))
    queries = serving_prefixes(data.test[:nq])
    print(f"[{stem}] {len(data.train)} sessions, {data.n_items} items, "
          f"{nq} queries; sections: {todo}")

    nbrs = None
    out_all = {"dataset": stem, "sections": {}}

    if "workload" in todo or "pareto" in todo:
        print("[d2] exact-Jaccard neighbors for the query stream...")
        t0 = time.perf_counter()
        nbrs = neighbor_ids(queries, data.train, args.k)
        print(f"  neighbor_ids: {time.perf_counter() - t0:.1f}s")

    if "workload" in todo:
        wl = run_workload(data, queries, nbrs, args.k, args.n_windows,
                          args.topm)
        out_all["sections"]["workload"] = wl
        if not args.out:
            write_json(config_out("d2", tags["workload"])[0],
                       {"dataset": stem, **wl})

    if "pareto" in todo:
        pa = run_pareto(nbrs, args.d, args.seed, args.sketch_w,
                        args.sketch_d)
        out_all["sections"]["pareto"] = pa
        if not args.out:
            write_json(config_out("d2", tags["pareto"])[0],
                       {"dataset": stem, **pa})

    if "ann" in todo:
        if not Path(ckpt).exists():
            print(f"[d2] ann SKIPPED: checkpoint not found: {ckpt}")
            out_all["sections"]["ann"] = f"skipped: no ckpt {ckpt}"
        else:
            print(f"[d2] ann on ckpt {ckpt.name} ...")
            common, methods = run_ann(data, queries, args.k, ckpt,
                                      args.ann_corpus_cap)
            out_all["sections"]["ann"] = {"common": common,
                                          "methods": methods}
            if not args.out:
                for mname, mrow in methods.items():
                    write_json(
                        config_out("d2",
                                   f"{tags['ann']}_{mname}")[0],
                        {"dataset": stem, **common,
                         "method": mname,
                         **({"result": mrow} if isinstance(mrow, dict)
                            else {"skipped": mrow})})

    if args.out:
        write_json(args.out, out_all)


if __name__ == "__main__":
    main()
