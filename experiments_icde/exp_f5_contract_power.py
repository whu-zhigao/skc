#!/usr/bin/env python3
"""F5: high-power bounded-dispersion calibration.

The earlier F1 probe used 500 queries on isolated checkpoints.  This stage
evaluates the operating checkpoint on a much larger leakage-free prefix set,
reports confidence intervals and hit counts, and includes an adversarial
profile perturbation derived from the task gradient.  It tests the consumer's
sensitivity to profile dispersion, not online admission.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REAL, config_out, load_data, load_model,  # noqa: E402
                    neighbor_ids, pool_profiles, run_forward,
                    serving_prefixes, write_json)
from mplignn.data import pad_batch  # noqa: E402


def default_ckpt(dataset: str) -> Path:
    for name in (f"abl_no_residual_{dataset}.pt", f"mplignn_{dataset}.pt"):
        path = REAL / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no checkpoint for {dataset}")


def evaluate(
    model,
    sessions,
    neighbors: np.ndarray,
    profiles: np.ndarray,
    device,
    batch_size: int,
    max_batches: int = 0,
) -> Dict[str, object]:
    hits = np.zeros(len(sessions), dtype=bool)
    reciprocal = np.zeros(len(sessions), dtype=np.float64)
    ranks_all = np.zeros(len(sessions), dtype=np.int64)
    end = len(sessions)
    if max_batches:
        end = min(end, max_batches * batch_size)
    for start in range(0, end, batch_size):
        stop = min(start + batch_size, end)
        chunk = sessions[start:stop]
        nb = neighbors[start:stop]
        seqs, _lens, targets = pad_batch(chunk, str(device))
        rows = profiles[np.asarray(nb, dtype=np.int64)]
        profile_tensor = __import__("torch").from_numpy(rows).to(device)
        logits = run_forward(model, seqs, profile_tensor).float()
        targets = targets.to(logits.device)
        ranks = (
            logits.argsort(dim=1, descending=True)
            == targets[:, None]
        ).nonzero(as_tuple=False)[:, 1] + 1
        hit_mask = (ranks <= 20).cpu().numpy()
        hits[start:stop] = hit_mask
        reciprocal[start:stop] = 0.0
        if hit_mask.any():
            reciprocal[start:stop][hit_mask] = (
                1.0 / ranks[ranks <= 20].cpu().numpy().astype(np.float64)
            )
        ranks_all[start:stop] = ranks.cpu().numpy()
    return {
        "n_evaluated": int(end),
        "hr20_hits": int(hits[:end].sum()),
        "HR@20": float(hits[:end].mean()),
        "MRR@20": float(reciprocal[:end].mean()),
        "mean_rank": float(ranks_all[:end].mean()),
        "_hits": hits[:end],
        "_reciprocal": reciprocal[:end],
    }


def paired_bootstrap_delta(
    baseline_hits: np.ndarray,
    treatment_hits: np.ndarray,
    baseline_rr: np.ndarray,
    treatment_rr: np.ndarray,
    iterations: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n = len(baseline_hits)
    hr_deltas = np.empty(iterations, dtype=np.float64)
    mrr_deltas = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        hr_deltas[i] = treatment_hits[idx].mean() - baseline_hits[idx].mean()
        mrr_deltas[i] = treatment_rr[idx].mean() - baseline_rr[idx].mean()
    return {
        "hr_delta": float(treatment_hits.mean() - baseline_hits.mean()),
        "mrr_delta": float(treatment_rr.mean() - baseline_rr.mean()),
        "hr_delta_ci95": [
            float(np.percentile(hr_deltas, 2.5)),
            float(np.percentile(hr_deltas, 97.5)),
        ],
        "mrr_delta_ci95": [
            float(np.percentile(mrr_deltas, 2.5)),
            float(np.percentile(mrr_deltas, 97.5)),
        ],
    }


def perturb_profiles(
    fresh: np.ndarray,
    active_ids: np.ndarray,
    drift: float,
    mode: str,
    seed: int,
) -> np.ndarray:
    stale = fresh.copy()
    if drift == 0 or not len(active_ids):
        return stale
    rng = np.random.default_rng(seed)
    if mode == "random":
        direction = rng.normal(
            size=(len(active_ids), fresh.shape[1])
        ).astype(np.float32)
        direction /= np.maximum(
            np.linalg.norm(direction, axis=1, keepdims=True), 1e-12
        )
        stale[active_ids] += drift * direction
    elif mode == "correlated":
        direction = rng.normal(size=fresh.shape[1]).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        stale[active_ids] += drift * direction[None, :]
    else:
        raise ValueError(f"unknown perturbation mode {mode}")
    realized = np.linalg.norm(stale[active_ids] - fresh[active_ids], axis=1)
    assert np.allclose(realized, drift, atol=2e-5)
    return stale


def adversarial_profiles(
    model,
    sessions,
    neighbors: np.ndarray,
    fresh: np.ndarray,
    active_ids: np.ndarray,
    drift: float,
    device,
    batch_size: int,
    max_batches: int,
    attack_loss: str = "ce",
) -> np.ndarray:
    """First-order task-loss attack within a per-profile L2 ball."""
    import torch
    import torch.nn.functional as F

    grad_sum = torch.zeros(fresh.shape, dtype=torch.float32, device=device)
    touched = torch.zeros(len(fresh), dtype=torch.float32, device=device)
    active_t = torch.as_tensor(active_ids, dtype=torch.long, device=device)
    end = len(sessions)
    if max_batches:
        end = min(end, max_batches * batch_size)
    for start in range(0, end, batch_size):
        stop = min(start + batch_size, end)
        chunk = sessions[start:stop]
        nb = neighbors[start:stop]
        seqs, _lens, targets = pad_batch(chunk, str(device))
        rows = fresh[np.asarray(nb, dtype=np.int64)]
        profile_tensor = torch.from_numpy(rows).to(device).requires_grad_(True)
        logits = run_forward(model, seqs, profile_tensor).float()
        targets = targets.to(device)
        if attack_loss == "ce":
            loss = F.cross_entropy(logits, targets)
        elif attack_loss == "margin":
            target_score = logits.gather(1, targets[:, None]).squeeze(1)
            masked = logits.clone()
            masked.scatter_(
                1, targets[:, None], float("-inf")
            )
            best_distractor = masked.max(dim=1).values
            loss = (best_distractor - target_score).mean()
        else:
            raise ValueError(f"unknown adversarial loss {attack_loss}")
        grad, = torch.autograd.grad(loss, profile_tensor)
        flat_ids = torch.from_numpy(
            np.asarray(nb, dtype=np.int64).reshape(-1)
        ).to(device)
        grad_sum.index_add_(0, flat_ids, grad.reshape(-1, fresh.shape[1]))
        touched.index_add_(
            0, flat_ids,
            torch.ones_like(flat_ids, dtype=torch.float32),
        )
    rows = torch.searchsorted(active_t, torch.nonzero(touched, as_tuple=False)[:, 0])
    selected_grad = grad_sum[active_t[rows]]
    selected_grad = selected_grad / selected_grad.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    stale = torch.from_numpy(fresh).to(device)
    stale[active_t[rows]] += drift * selected_grad
    return stale.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--n-query", type=int, default=5000,
                    help="0 uses the complete test split")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--drifts", default="0.025,0.05,0.1,0.2,0.4,0.8")
    ap.add_argument("--fractions", default="1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--modes", default="random,correlated,adversarial")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="smoke-test cap")
    ap.add_argument("--gradient-fraction", type=float, default=0.5,
                    help="initial query fraction used to fit the adversarial direction")
    ap.add_argument("--adversarial-loss", choices=["ce", "margin"],
                    default="ce")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    stem = args.data.stem
    ckpt = args.ckpt or default_ckpt(stem)
    n_query = args.n_query or 0
    tag = (
        f"{stem}_power_heldout_g{args.gradient_fraction:g}_"
        f"nq{n_query if n_query else 'all'}_bs{args.batch_size}"
    )
    out_path, skip = config_out("f5", tag, args.out, args.force)
    if skip:
        print(f"[f5] exists, skipping: {out_path} (--force to rerun)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data)
    n_query = min(n_query or len(data.test), len(data.test))
    sessions = data.test[:n_query]
    queries = serving_prefixes(sessions)
    neighbors = np.asarray(
        neighbor_ids(queries, data.train, 10), dtype=np.int64
    )
    # Guarantee exactly ten neighbor slots for tensor reshaping.
    if neighbors.ndim != 2 or neighbors.shape[1] != 10:
        fixed = np.zeros((len(queries), 10), dtype=np.int64)
        for i, row in enumerate(neighbors):
            reps = (10 + len(row) - 1) // max(1, len(row))
            fixed[i] = np.tile(row, reps)[:10]
        neighbors = fixed

    split = int(len(sessions) * args.gradient_fraction)
    split = min(max(split, 1), max(1, len(sessions) - 1))
    gradient_sessions = sessions[:split]
    gradient_neighbors = neighbors[:split]
    eval_sessions = sessions[split:]
    eval_neighbors = neighbors[split:]

    model = load_model(data, ckpt, device=str(device))
    emb = model.embedding.weight.detach().cpu().numpy()
    fresh = pool_profiles(emb, data.train)
    active_ids = np.unique(neighbors.reshape(-1))
    baseline = evaluate(
        model, eval_sessions, eval_neighbors, fresh, device,
        args.batch_size, args.max_batches,
    )
    baseline_public = {k: v for k, v in baseline.items() if not k.startswith("_")}

    drifts = [float(x) for x in args.drifts.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    modes = [x.strip() for x in args.modes.split(",")]
    points = []

    for mode in modes:
        for drift in drifts:
            all_stale = None
            if mode in ("adversarial", "margin"):
                attack_loss = "ce" if mode == "adversarial" else "margin"
                all_stale = adversarial_profiles(
                    model, gradient_sessions, gradient_neighbors,
                    fresh, active_ids,
                    drift, device, args.batch_size, args.max_batches,
                    attack_loss=attack_loss,
                )
            for fraction in fractions:
                # Replicate the seed at the grid level so each fraction sees
                # the same perturbation family but independent row choices.
                metric_seeds = []
                mode_seeds = seeds if mode == "random" else [0]
                for seed in mode_seeds:
                    if mode in ("adversarial", "margin"):
                        stale_all = all_stale
                    else:
                        stale_all = perturb_profiles(
                            fresh, active_ids, drift, mode, seed
                        )
                    stale = fresh.copy()
                    count = int(round(fraction * eval_neighbors.shape[1]))
                    if count:
                        rng = np.random.default_rng(10_000 + seed)
                        cols = rng.permuted(
                            np.tile(
                                np.arange(eval_neighbors.shape[1]),
                                (len(eval_neighbors), 1),
                            ),
                            axis=1,
                        )[:, :count]
                        rows_idx = np.repeat(
                            np.arange(len(eval_neighbors)), count
                        )
                        chosen = eval_neighbors[rows_idx, cols.reshape(-1)]
                        stale[chosen] = stale_all[chosen]
                    ev = evaluate(
                        model, eval_sessions, eval_neighbors, stale, device,
                        args.batch_size, args.max_batches,
                    )
                    realized = np.linalg.norm(
                        stale[active_ids] - fresh[active_ids], axis=1
                    )
                    nonzero = realized[realized > 0]
                    stats = paired_bootstrap_delta(
                        baseline["_hits"], ev["_hits"],
                        baseline["_reciprocal"], ev["_reciprocal"],
                        args.bootstrap, seed,
                    )
                    points.append({
                        "mode": mode,
                        "drift_l2": drift,
                        "stale_fraction": fraction,
                        "seed": seed,
                        "realized_dispersion": {
                            "perturbed_rows": int(len(nonzero)),
                            "mean": float(np.mean(nonzero)) if len(nonzero) else 0.0,
                            "p99": float(np.percentile(nonzero, 99))
                            if len(nonzero) else 0.0,
                            "max": float(nonzero.max(initial=0.0)),
                        },
                        **{k: v for k, v in ev.items() if not k.startswith("_")},
                        **stats,
                    })
                    print(
                        f"{mode} delta={drift:.3f} fraction={fraction:.2f} "
                        f"seed={seed}: HR20={ev['HR@20']:.4f} "
                        f"delta={stats['hr_delta']:+.4f}", flush=True,
                    )

    result = {
        "dataset": stem,
        "checkpoint": str(ckpt),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0)
        if device.type == "cuda" else None,
        "protocol": (
            "leakage-free prefixes; paired bootstrap; random/correlated "
            "perturbations; task-gradient adversarial direction fitted on "
            "the initial query split and evaluated on the held-out split"
        ),
        "gradient_split_queries": split,
        "heldout_eval_queries": len(eval_sessions),
        "n_queries_requested": len(sessions),
        "batch_size": args.batch_size,
        "bootstrap_iterations": args.bootstrap,
        "baseline": baseline_public,
        "active_unique_profiles": int(len(active_ids)),
        "points": points,
    }
    write_json(out_path, result)


if __name__ == "__main__":
    main()
