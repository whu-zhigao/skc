#!/usr/bin/env python3
"""F4: one integrated hard-cap SKC operating point.

Unlike the older packed-page and cold-tier microbenchmarks, this stage uses
one corpus, one leakage-free query replay, real checkpoint-derived profiles,
int8 packed pages, and a posting tier that contains no duplicate profiles.
Soft misses and refreshes recompute profiles from the live embedding table;
no full source-profile matrix is kept resident during serving.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS, REAL, config_out, load_data, load_model,  # noqa: E402
                    pool_profiles, run_forward, serving_prefixes, write_json)
from mplignn.data import pad_batch  # noqa: E402
from systems.bounded_tier import build_bounded_tier  # noqa: E402
from systems.hot_pages import BoundedPackedPageStore  # noqa: E402
from systems.igms import ExactJaccard  # noqa: E402


def maxrss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    if sys.platform == "darwin":
        return value / (1024.0 ** 2)
    return value / 1024.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def default_ckpt(dataset: str) -> Path:
    for name in (
        f"abl_no_residual_{dataset}.pt",
        f"mplignn_{dataset}.pt",
    ):
        path = REAL / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no checkpoint for {dataset}")


def exact_truth(data, queries: List[List[int]], k: int, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache, allow_pickle=False)
    index = ExactJaccard()
    for session in data.train:
        index.insert(session)
    index.finalize()
    truth: List[List[int]] = []
    start = time.perf_counter()
    for i, query in enumerate(queries):
        row = index.topk(query, k)
        if len(row) < k:
            reps = (k + len(row) - 1) // max(1, len(row))
            row = np.tile(row, reps)[:k].tolist()
        truth.append(row)
        if (i + 1) % 250 == 0:
            print(f"  exact truth {i + 1}/{len(queries)}", flush=True)
    arr = np.asarray(truth, dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    print(f"exact truth built in {time.perf_counter() - start:.2f}s")
    return arr


def make_source_fn(embeddings: np.ndarray, sessions):
    stats = {"calls": 0, "rows": 0, "seconds": 0.0}

    def recompute(source_ids: Sequence[int], *, count: bool = True) -> np.ndarray:
        t0 = time.perf_counter()
        ids = np.asarray(source_ids, dtype=np.int64)
        out = np.empty((len(ids), embeddings.shape[1]), dtype=np.float32)
        for row, sid in enumerate(ids.tolist()):
            items = np.asarray(sorted(set(sessions[int(sid)])), dtype=np.int64)
            out[row] = embeddings[items].mean(axis=0, dtype=np.float32)
        if count:
            stats["calls"] += 1
            stats["rows"] += int(len(ids))
            stats["seconds"] += time.perf_counter() - t0
        return out

    recompute.stats = stats
    recompute.embeddings = embeddings
    recompute.sessions = sessions
    return recompute, stats


def recall_at_k(pred: Sequence[int], truth: Sequence[int], k: int) -> float:
    p = set(int(x) for x in pred[:k])
    t = set(int(x) for x in truth[:k])
    return len(p & t) / max(1, min(k, len(t)))


def evaluate_replay(
    data,
    queries: List[List[int]],
    truth: np.ndarray,
    posting,
    packed: BoundedPackedPageStore,
    source_fn,
    k: int,
    predictions: Optional[List[List[int]]] = None,
) -> dict:
    posting_ms: List[float] = []
    gather_ms: List[float] = []
    recalls: List[float] = []
    profile_hits: List[float] = []
    actual_errors: List[float] = []
    served_bounds: List[float] = []
    source_stats = source_fn.stats
    source_start = int(source_stats.get("rows", 0))

    for i, (query, truth_ids) in enumerate(zip(queries, truth)):
        if predictions is None:
            t0 = time.perf_counter()
            pred = posting.search(query, return_profiles=False)
            t1 = time.perf_counter()
        else:
            pred = predictions[i]
            t0 = t1 = time.perf_counter()
        values, hit, bound = packed.service(
            pred, fallback_fn=source_fn, strict_dirty=False
        )
        t2 = time.perf_counter()
        posting_ms.append((t1 - t0) * 1000.0)
        gather_ms.append((t2 - t1) * 1000.0)
        recalls.append(recall_at_k(pred, truth_ids, k))
        if pred:
            profile_hits.append(float(hit))
        served_bounds.extend([float(bound)] * len(pred))

        # Every returned row (cached or recomputed) is compared with the row
        # implied by the current table. Dirty/missing rows are exact, while
        # clean cached rows must lie inside the reported page bound.
        fresh = source_fn(pred, count=False)
        actual = np.linalg.norm(values - fresh, axis=1)
        actual_errors.extend(float(x) for x in actual)

    serving_source_rows = int(source_stats.get("rows", 0)) - source_start
    actual = np.asarray(actual_errors, dtype=np.float64)
    bounds = np.asarray(served_bounds, dtype=np.float64)
    expanded_bounds = bounds[: len(actual)]
    violations = int(np.sum(actual > expanded_bounds + 3e-6))
    return {
        "posting_recall_at_10": float(np.mean(recalls)),
        "packed_profile_hit_rate": float(np.mean(profile_hits)),
        "posting_ms": None if predictions is not None else {
            "p50": percentile(posting_ms, 50),
            "p90": percentile(posting_ms, 90),
            "p99": percentile(posting_ms, 99),
        },
        "gather_ms": {
            "p50": percentile(gather_ms, 50),
            "p90": percentile(gather_ms, 90),
            "p99": percentile(gather_ms, 99),
        },
        "serving_source_recompute_rows": serving_source_rows,
        "actual_profile_dispersion_max": float(actual.max(initial=0.0)),
        "reported_bound_max": float(bounds.max(initial=0.0)),
        "contract_violations": violations,
        "contract_checks": int(len(actual)),
    }


def perturb_embeddings(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    drift: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Move selected rows by exact L2 drift along deterministic directions."""
    out = embeddings.copy()
    if not len(item_ids):
        return out
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(len(item_ids), embeddings.shape[1])).astype(np.float32)
    direction /= np.maximum(
        np.linalg.norm(direction, axis=1, keepdims=True), 1e-12
    )
    out[item_ids] += (drift.astype(np.float32)[:, None] * direction)
    return out


def ranking_metrics(model, sessions, neighbor_rows, profile_fn, device, k,
                    batch_size: int = 256) -> dict:
    """End-to-end ranking audit with leakage-free prefixes."""
    import torch

    hits = 0
    reciprocal = 0.0
    for start in range(0, len(sessions), batch_size):
        stop = min(start + batch_size, len(sessions))
        chunk = sessions[start:stop]
        rows = []
        for i in range(start, stop):
            row = [int(sid) for sid in neighbor_rows[i]]
            if not row:
                row = [0] * k
            reps = (k + len(row) - 1) // max(1, len(row))
            rows.append((row * reps)[:k])
        profiles = np.stack([
            profile_fn([int(sid) for sid in row]) for row in rows
        ])
        seqs, _lens, targets = pad_batch(chunk, str(device))
        logits = run_forward(
            model, seqs, torch.from_numpy(profiles).to(device)
        ).float()
        targets = targets.to(logits.device)
        ranks = (
            logits.argsort(dim=1, descending=True) == targets[:, None]
        ).nonzero(as_tuple=False)[:, 1] + 1
        mask = (ranks <= 20).cpu().numpy()
        hits += int(mask.sum())
        rr = np.zeros(len(chunk), dtype=np.float64)
        rr[mask] = 1.0 / ranks[ranks <= 20].cpu().numpy().astype(np.float64)
        reciprocal += float(rr.sum())
    n = len(sessions)
    return {
        "n_queries": n,
        "HR@20": hits / n,
        "MRR@20": reciprocal / n,
        "hr20_hits": hits,
    }


def projection_audit(model, old_embeddings, new_embeddings, sessions,
                     returned_sessions, drifted_items, drift_values) -> dict:
    """Check the operator-norm bound for active fixed-weight projections."""
    active_items = sorted({
        int(item)
        for sid in returned_sessions
        for item in sessions[int(sid)]
    })
    drift_by_item = {
        int(item): float(value)
        for item, value in zip(drifted_items, drift_values)
    }
    projections = []
    operator_norms = []
    for layer_name in ("wq", "wk", "wv"):
        layer = getattr(model.global_, layer_name, None)
        if layer is None:
            return {
                "status": "unavailable",
                "missing_layer": layer_name,
            }
        weight = layer.weight.detach().cpu().numpy()
        operator_norms.append(float(np.linalg.norm(weight, ord=2)))
        old = old_embeddings[active_items] @ weight.T
        new = new_embeddings[active_items] @ weight.T
        projections.append((old, new))

    actual_max = 0.0
    bound_max = 0.0
    checks = 0
    violations = 0
    max_operator = max(operator_norms)
    for old, new in projections:
        actual = np.linalg.norm(new - old, axis=1)
        bound = np.asarray([
            max_operator * drift_by_item.get(int(item), 0.0)
            for item in active_items
        ])
        actual_max = max(actual_max, float(actual.max(initial=0.0)))
        bound_max = max(bound_max, float(bound.max(initial=0.0)))
        checks += int(len(actual))
        violations += int(np.sum(actual > bound + 2e-6))
    return {
        "status": "ok",
        "active_items": len(active_items),
        "operator_norms_q_k_v": operator_norms,
        "fp32_projection_cache_mb": (
            len(active_items) * old_embeddings.shape[1] * 3 * 4 / 1e6
        ),
        "actual_max_projection_drift": actual_max,
        "operator_norm_bound_max": bound_max,
        "checks": checks,
        "violations": violations,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--n-query", type=int, default=3000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--packed-budget-mb", type=float, default=5.0)
    ap.add_argument("--posting-budget-mb", type=float, default=8.0)
    ap.add_argument("--block-sessions", type=int, default=8192,
                    help="hard-cap posting block size")
    ap.add_argument("--bands", type=int, default=32)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--drift-items", type=int, default=80)
    ap.add_argument("--drift", type=float, default=0.15)
    ap.add_argument("--drift-seed", type=int, default=917)
    ap.add_argument("--quality-batch-size", type=int, default=256)
    ap.add_argument("--evict-after-read", action="store_true")
    ap.add_argument("--keep-build", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    stem = args.data.stem
    ckpt = args.ckpt or default_ckpt(stem)
    n_query = min(args.n_query, len(load_data(args.data).test))
    tag = (f"{stem}_integrated_nq{n_query}_p{args.packed_budget_mb:g}_"
           f"i{args.posting_budget_mb:g}_theta{args.theta:g}_s{args.drift_seed}")
    out_path, skip = config_out("f4", tag, args.out, args.force)
    if skip:
        print(f"[f4] exists, skipping: {out_path} (--force to rerun)")
        return

    data = load_data(args.data)
    sessions = data.test[:n_query]
    queries = serving_prefixes(sessions)
    truth = exact_truth(
        data, queries, args.k,
        RESULTS / "f4" / f"{stem}_exact_top{args.k}_nq{n_query}.npy",
    )

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(data, ckpt, device=str(device))
    original_embeddings = model.embedding.weight.detach().cpu().numpy().copy()
    block_sessions = int(args.block_sessions)
    root = Path("/tmp") / f"skc_f4_{stem}_{args.drift_seed}"
    if root.exists():
        shutil.rmtree(root)
    t0 = time.perf_counter()
    # profiles=None is intentional: the posting tier carries no duplicate
    # model-valued payload. Profiles are gathered only from packed pages or
    # recomputed from the embedding table.
    posting = build_bounded_tier(
        data.train,
        None,
        root,
        block_sessions=block_sessions,
        bands=args.bands,
        rows=args.rows,
        hot_blocks=1,
    )
    posting.memory_budget_bytes = int(args.posting_budget_mb * 1e6)
    posting.enforce_budget()
    posting.evict_after_read = args.evict_after_read
    posting_build_s = time.perf_counter() - t0

    # First execute the retrieval plane once.  The packed resident set is then
    # profiled from the neighbor ids this substrate actually returns, rather
    # than from exact-truth ids that the serving path never observes.
    predictions: List[List[int]] = []
    posting_latencies: List[float] = []
    posting_recalls: List[float] = []
    resident_max_during_posting = 0
    for query, truth_ids in zip(queries, truth):
        t0 = time.perf_counter()
        pred = posting.search(query, return_profiles=False)
        posting_latencies.append((time.perf_counter() - t0) * 1000.0)
        predictions.append([int(x) for x in pred])
        posting_recalls.append(recall_at_k(pred, truth_ids, args.k))
        resident_max_during_posting = max(
            resident_max_during_posting, posting.resident_bytes
        )
    posting_replay = {
        "recall_at_10": float(np.mean(posting_recalls)),
        "p50_ms": percentile(posting_latencies, 50),
        "p90_ms": percentile(posting_latencies, 90),
        "p99_ms": percentile(posting_latencies, 99),
        "resident_max_mb": resident_max_during_posting / 1e6,
        "unique_returned_sessions": int(
            len({sid for pred in predictions for sid in pred})
        ),
    }

    t0 = time.perf_counter()
    build_profiles = pool_profiles(original_embeddings, data.train)
    profile_build_s = time.perf_counter() - t0
    flat_predictions = np.concatenate(
        [np.asarray(x, dtype=np.int64) for x in predictions]
    )
    frequencies = np.bincount(flat_predictions, minlength=len(data.train))
    candidate_order = np.flatnonzero(frequencies > 0).astype(np.int64)
    candidate_order = candidate_order[
        np.argsort(-frequencies[candidate_order], kind="stable")
    ]
    packed = BoundedPackedPageStore(
        int(args.packed_budget_mb * 1e6),
        dimension=original_embeddings.shape[1],
        page_size=128,
        theta_max=args.theta,
        quantize=True,
        quant_bits=8,
    )
    packed.admit(candidate_order.tolist(), build_profiles, data.train)
    del build_profiles
    gc.collect()

    source_fn, source_stats = make_source_fn(original_embeddings, data.train)
    initial = evaluate_replay(
        data, queries, truth, posting, packed, source_fn, args.k,
        predictions=predictions,
    )
    initial_source_stats = dict(source_stats)

    exact_quality = ranking_metrics(
        model, sessions, truth,
        lambda ids: source_fn(ids, count=False),
        device,
        batch_size=args.quality_batch_size, k=args.k,
    )
    posting_fresh_quality = ranking_metrics(
        model, sessions, predictions,
        lambda ids: source_fn(ids, count=False),
        device,
        batch_size=args.quality_batch_size, k=args.k,
    )
    posting_cached_quality = ranking_metrics(
        model, sessions, predictions,
        lambda ids: packed.service(
            ids, fallback_fn=lambda missing: source_fn(missing, count=False),
            strict_dirty=False,
        )[0],
        device,
        batch_size=args.quality_batch_size, k=args.k,
    )

    # Controlled drift stress: the query side uses the current table; stale
    # packed rows keep their old values until a page crosses the bound.
    rng = np.random.default_rng(args.drift_seed)
    n_items = min(args.drift_items, original_embeddings.shape[0])
    drifted_items = rng.choice(original_embeddings.shape[0], size=n_items,
                               replace=False).astype(np.int64)
    drift_values = np.full(n_items, args.drift, dtype=np.float64)
    current_embeddings = perturb_embeddings(
        original_embeddings, drifted_items, drift_values, args.drift_seed + 1
    )
    drift_source_fn, drift_source_stats = make_source_fn(
        current_embeddings, data.train
    )
    projection_result = projection_audit(
        model,
        original_embeddings,
        current_embeddings,
        data.train,
        {sid for pred in predictions for sid in pred},
        drifted_items,
        drift_values,
    )
    # The old table and its closure are no longer needed after the controlled
    # perturbation; keep only the current maintenance/source table resident.
    del source_fn, original_embeddings
    gc.collect()
    packed.update_drift(
        {int(item): float(value) for item, value in zip(drifted_items, drift_values)}
    )
    dirty_after_drift = len(packed.dirty_pages())
    dirty_replay = evaluate_replay(
        data, queries, truth, posting, packed, drift_source_fn, args.k,
        predictions=predictions,
    )

    t0 = time.perf_counter()
    refreshed_pages = packed.refresh(source_fn=drift_source_fn)
    dai2_refresh_s = time.perf_counter() - t0
    still_dirty = len(packed.dirty_pages())
    refreshed_replay = evaluate_replay(
        data, queries, truth, posting, packed, drift_source_fn, args.k,
        predictions=predictions,
    )

    # Full invalidation comparison after resetting all page ledgers.
    packed.mark_all_dirty()
    t0 = time.perf_counter()
    full_refresh_pages = packed.refresh(source_fn=drift_source_fn)
    full_refresh_s = time.perf_counter() - t0

    cold_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    result = {
        "dataset": stem,
        "checkpoint": str(ckpt),
        "protocol": (
            "one integrated corpus/query replay; real checkpoint profiles; "
            "int8 packed pages; posting tier stores no profiles; source "
            "profiles recomputed from the live embedding table"
        ),
        "admission": (
            "full-window NFA over neighbor ids returned by the serving posting "
            "substrate; profiled ceiling, not an online prediction claim"
        ),
        "n_sessions": len(data.train),
        "n_queries": n_query,
        "k": args.k,
        "block_sessions": block_sessions,
        "bands": args.bands,
        "rows": args.rows,
        "theta_max": args.theta,
        "quant_bits": 8,
        "source_mode": "recompute-from-embedding",
        "source_embedding_table_mb": current_embeddings.nbytes / 1e6,
        "profile_build_seconds": profile_build_s,
        "posting_build_seconds": posting_build_s,
        "packed": {
            "budget_mb": args.packed_budget_mb,
            "resident_mb": packed.resident_bytes() / 1e6,
            "admitted_sessions": (
                0 if packed.resident_sids is None else len(packed.resident_sids)
            ),
            "pages": len(packed.pages),
            "prediction_sessions_not_packed": int(
                len(
                    {sid for pred in predictions for sid in pred}
                    - set(
                        [] if packed.resident_sids is None
                        else packed.resident_sids.tolist()
                    )
                )
            ),
            "prediction_membership_hit_rate": float(
                np.mean([
                    sid in (
                        set() if packed.resident_sids is None
                        else set(packed.resident_sids.tolist())
                    )
                    for pred in predictions for sid in pred
                ])
            ),
            "resident_components_mb": {
                k: v / 1e6 for k, v in packed.resident_components().items()
            },
            "contract_violations": (
                initial["contract_violations"]
                + dirty_replay["contract_violations"]
                + refreshed_replay["contract_violations"]
            ),
        },
        "posting": {
            "budget_mb": args.posting_budget_mb,
            "resident_end_mb": posting.resident_bytes / 1e6,
            "temporary_max_mb": posting.temporary_bytes_max / 1e6,
            "cold_storage_mb": cold_bytes / 1e6,
            "disk_bucket_reads": posting.disk_bucket_reads,
            "disk_item_reads": posting.disk_item_reads,
            "disk_bucket_bytes_mb": posting.disk_bucket_bytes / 1e6,
            "disk_item_bytes_mb": posting.disk_item_bytes / 1e6,
            "evict_after_read": bool(args.evict_after_read),
        },
        "posting_replay": posting_replay,
        "initial": initial,
        "tier2_projection_audit": projection_result,
        "quality": {
            "exact_neighbors_fresh": exact_quality,
            "posting_neighbors_fresh": posting_fresh_quality,
            "posting_neighbors_int8_cached": posting_cached_quality,
        },
        "drift": {
            "items": int(n_items),
            "l2": args.drift,
            "dirty_pages": dirty_after_drift,
            "replay": dirty_replay,
            "dai2_refresh_pages": refreshed_pages,
            "dai2_refresh_seconds": dai2_refresh_s,
            "still_dirty_after_refresh": still_dirty,
            "refreshed_replay": refreshed_replay,
            "full_invalidation_pages": full_refresh_pages,
            "full_invalidation_seconds": full_refresh_s,
        },
        "source_recompute_stats": {
            "initial": initial_source_stats,
            "drift": drift_source_stats,
        },
        "combined_resident_end_mb": (
            packed.resident_bytes() + posting.resident_bytes
        ) / 1e6,
        "combined_peak_resident_mb": (
            packed.resident_bytes()
            + posting_replay["resident_max_mb"] * 1e6
            + posting.temporary_bytes_max
        ) / 1e6,
        "combined_resident_plus_peak_temp_mb": (
            packed.resident_bytes()
            + posting_replay["resident_max_mb"] * 1e6
            + posting.temporary_bytes_max
        ) / 1e6,
        "maintenance_plane_source_table_mb": current_embeddings.nbytes / 1e6,
        "process_maxrss_mb": maxrss_mb(),
    }
    if not args.keep_build:
        shutil.rmtree(root, ignore_errors=True)
    write_json(out_path, result)


if __name__ == "__main__":
    main()
